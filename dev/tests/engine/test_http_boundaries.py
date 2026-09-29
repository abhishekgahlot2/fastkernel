import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

from PIL import Image

from dev.tests.test_server import FakeRuntime, Harness
from server import crash_trace, http_security, images
from server.errors import APIError


class HttpBoundaryTests(unittest.TestCase):
    def test_model_retrieval_matches_listing_without_inference(self):
        harness = Harness(FakeRuntime(), model="community/example-model")
        self.addCleanup(harness.close)
        status, _, body = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        model = json.loads(body)["data"][0]
        for suffix in (
            "community/example-model",
            "community%2Fexample-model?beta=true",
        ):
            with self.subTest(suffix=suffix):
                status, _, body = harness.request("GET", "/v1/models/" + suffix)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), model)
        for suffix in ("", "community/other-model", "community/example-model/extra"):
            with self.subTest(suffix=suffix):
                status, _, body = harness.request("GET", "/v1/models/" + suffix)
                self.assertEqual(status, 404)
                self.assertEqual(json.loads(body)["error"]["code"], "model_not_found")
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_authority_and_origin_validation(self):
        allowed = {"localhost", "127.0.0.1", "::1", "serving.example"}
        cases = (
            ("localhost:8000", "http://localhost:8000", True),
            ("[::1]:8000", "http://[::1]:8000", True),
            ("serving.example", "https://serving.example", True),
            ("127.0.0.1:8000", None, True),
            ("unconfigured.example:8000", None, False),
            ("localhost:8000", "http://localhost:9000", False),
            ("localhost:8000", "http://other.example:8000", False),
            ("localhost:8000", "null", False),
            ("localhost:8000", "http://user@localhost:8000", False),
            ("localhost:8000", "http://localhost:8000/path", False),
        )
        for host, origin, valid in cases:
            with self.subTest(host=host, origin=origin):
                headers = Message()
                headers["Host"] = host
                if origin is not None:
                    headers["Origin"] = origin
                if valid:
                    http_security.validate_headers(headers, allowed)
                else:
                    with self.assertRaises(APIError):
                        http_security.validate_headers(headers, allowed)
        for name in ("Host", "Origin"):
            headers = Message()
            headers["Host"] = "localhost"
            headers["Origin"] = "http://localhost"
            headers[name] = headers[name]
            with self.assertRaises(APIError):
                http_security.validate_headers(headers, allowed)

    def test_http_rejection_precedes_routing_and_local_access_still_works(self):
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)
        status, _, body = harness.request(
            "GET", "/health", headers={"Host": "unconfigured.example"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["error"]["code"], "forbidden")
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_each_request_gets_an_independent_public_id(self):
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        }
        with mock.patch(
            "server.server.secrets.token_hex",
            side_effect=("first-random-id", "second-random-id"),
        ):
            first, _, _ = harness.app.prepare(body)
            second, _, _ = harness.app.prepare(body)
        self.assertEqual(
            (first.public_id, second.public_id), ("first-random-id", "second-random-id")
        )
        self.assertNotEqual(first.request_id, second.request_id)

    def test_image_formats_are_explicit(self):
        for format_name in ("PNG", "JPEG", "WEBP", "GIF", "BMP"):
            with self.subTest(format=format_name):
                data = io.BytesIO()
                Image.new("RGB", (32, 32), "red").save(data, format=format_name)
                if format_name in images.IMAGE_FORMATS:
                    self.assertGreater(images.prepare(data.getvalue()).tokens, 0)
                else:
                    with self.assertRaises(images.ImageError):
                        images.prepare(data.getvalue())

    def test_crash_content_is_disabled_without_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory):
                ring = crash_trace.CrashTraceRing(("splash", "serve-native"))
                ring.start_generation(1, 1)
                ring.record_bytes(1, "client_to_engine", b"private synthetic content")
                self.assertIsNone(
                    ring.dump(
                        1,
                        RuntimeError("failed"),
                        process_returncode=1,
                        last_status=None,
                    )
                )
                self.assertEqual(list(directory.iterdir()), [])

    def test_opt_in_crash_files_have_a_total_retention_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory),
                mock.patch.object(crash_trace, "MAX_TRACE_FILES", 2),
            ):
                ring = crash_trace.CrashTraceRing(
                    ("splash", "serve-native"), enabled=True
                )
                for generation in range(1, 5):
                    ring.start_generation(generation, 1)
                    ring.record_bytes(generation, "client_to_engine", b"synthetic")
                    ring.dump(
                        generation,
                        RuntimeError("failed"),
                        process_returncode=1,
                        last_status=None,
                    )
                files = list(directory.glob("*.json"))
                self.assertEqual(len(files), 2)
                self.assertEqual(
                    {json.loads(path.read_text())["generation"] for path in files},
                    {3, 4},
                )


if __name__ == "__main__":
    unittest.main()
