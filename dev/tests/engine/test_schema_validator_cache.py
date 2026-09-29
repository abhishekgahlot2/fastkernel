import gc
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from jsonschema import SchemaError
from referencing import Registry

from server import schema_validation as validation
from server import tool_schema


class ValidatorCacheTests(unittest.TestCase):
    def setUp(self):
        self.clear()
        self.addCleanup(self.clear)

    @staticmethod
    def clear():
        with validation._validator_cache_lock:
            validation._validator_cache.clear()
            validation._validator_cache_bytes = 0

    def build(self, schema):
        return validation.build_validator(
            schema, tool_schema._schemas, tool_schema.LOCAL_REGISTRY
        )

    def test_eviction_obeys_both_count_and_source_byte_limits(self):
        for count, budget in ((2, 10000), (256, 180)):
            with (
                self.subTest(count=count, budget=budget),
                mock.patch.object(validation, "_VALIDATOR_CACHE_SIZE", count),
                mock.patch.object(validation, "_VALIDATOR_CACHE_SOURCE_BYTES", budget),
            ):
                self.clear()
                for i in range(20):
                    self.build({"type": "string", "description": str(i) + "x" * 30})
                    cache = validation._validator_cache
                    self.assertLessEqual(len(cache), count)
                    self.assertLessEqual(validation._validator_cache_bytes, budget)
                    self.assertEqual(
                        validation._validator_cache_bytes,
                        sum(len(key[2]) for key in cache),
                    )

    def test_cache_hit_refreshes_lru(self):
        with mock.patch.object(validation, "_VALIDATOR_CACHE_SIZE", 2):
            a, b, c = ({"const": i} for i in range(3))
            first = self.build(a)
            second = self.build(b)
            self.assertIs(first, self.build(a))
            self.build(c)
            self.assertIs(first, self.build(a))
            self.assertIsNot(second, self.build(b))

    def test_oversized_schema_is_validated_without_retention(self):
        with mock.patch.object(validation, "_VALIDATOR_CACHE_SOURCE_BYTES", 100):
            schema = {"type": "integer", "description": "x" * 200}
            first = self.build(schema)
            self.assertTrue(first.is_valid(1))
            self.assertFalse(first.is_valid("1"))
            self.assertIsNot(first, self.build(schema))
            self.assertEqual(validation._validator_cache_bytes, 0)
            self.assertFalse(validation._validator_cache)
            with self.assertRaises(SchemaError):
                self.build({**schema, "type": "invalid"})

    def test_invalid_schema_is_not_cached(self):
        for _ in range(2):
            with self.assertRaises(SchemaError):
                self.build({"type": "invalid"})
        self.assertFalse(validation._validator_cache)
        self.assertEqual(validation._validator_cache_bytes, 0)

    def test_caller_mutation_does_not_change_cached_validator(self):
        schema = {"properties": {"x": {"type": "integer"}}}
        first = self.build(schema)
        schema["properties"]["x"]["type"] = "string"
        second = self.build(schema)
        self.assertTrue(first.is_valid({"x": 1}))
        self.assertFalse(first.is_valid({"x": "1"}))
        self.assertFalse(second.is_valid({"x": 1}))
        self.assertTrue(second.is_valid({"x": "1"}))

    def test_identity_contexts_stay_alive_until_eviction(self):
        def nodes(schema):
            return [schema]

        ref = weakref.ref(nodes)
        registry = Registry()
        schema = {"type": "integer"}
        first = validation.build_validator(schema, nodes, registry)
        self.assertIsNot(
            first, validation.build_validator(schema, lambda s: [s], registry)
        )
        del nodes
        gc.collect()
        self.assertIsNotNone(ref())
        self.clear()
        gc.collect()
        self.assertIsNone(ref())

    def test_concurrent_misses_account_for_one_entry(self):
        barrier = threading.Barrier(4)
        bounded_class = validation._bounded_class

        def delayed_class(base):
            barrier.wait(timeout=5)
            return bounded_class(base)

        with (
            mock.patch.object(validation, "_bounded_class", delayed_class),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            results = list(pool.map(self.build, [{"type": "integer"}] * 4))
        self.assertTrue(all(v is results[0] for v in results))
        self.assertEqual(len(validation._validator_cache), 1)
        self.assertEqual(
            validation._validator_cache_bytes,
            sum(len(key[2]) for key in validation._validator_cache),
        )
