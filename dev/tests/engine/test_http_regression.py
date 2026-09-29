import contextlib
import copy
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.benchmarks import http_regression as benchmark
from dev.tests import smoke_real as smoke


class CharacterTokenizer:
    def encode(self, content, **_kwargs):
        return list(content)

    def decode(self, tokens):
        return "".join(tokens)

    def apply_chat_template(self, messages, **_kwargs):
        return self.encode("<user>" + messages[0]["content"] + "<assistant>")


class HttpRegressionTests(unittest.TestCase):
    def test_any_repository_selects_matching_package_and_api_name(self):
        for selected in (
            "incoai/Qwen3.8-27B-Splash",
            "incoai/Qwen3.6-35B-A3B-Splash",
            "community/custom-splash",
        ):
            with self.subTest(model=selected):
                arguments = smoke.parse_args(["--model", selected])
                self.assertEqual(arguments.model, selected)
                self.assertEqual(
                    arguments.package,
                    smoke.model_artifacts.installed_root(
                        smoke.model_artifacts.MODELS, selected
                    ),
                )
        model = "incoai/Qwen3.8-27B-Splash"
        arguments = smoke.parse_args(["--package", "custom-package", "--model", model])
        self.assertEqual(str(arguments.package), "custom-package")
        self.assertEqual(arguments.model, model)

    def test_real_helpers_require_a_canonical_model(self):
        for parse in (smoke.parse_args, benchmark.parse_args):
            for arguments in ([], ["--model", "qwen3.8-27b"], ["--model", "custom"]):
                with (
                    self.subTest(parse=parse, arguments=arguments),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        parse(arguments)

    def test_benchmark_resolves_the_selected_model_package(self):
        model = "incoai/Qwen3.6-35B-A3B-Splash"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "splash"
            binary.touch()
            (root / "splash.metallib").touch()
            package = root / "models" / model
            (package / "tokenizer").mkdir(parents=True)
            (package / "manifest.json").write_text("{}")
            with mock.patch.object(smoke.model_artifacts, "MODELS", root / "models"):
                arguments = benchmark.parse_args(
                    [
                        "--model",
                        model,
                        "--binary",
                        str(binary),
                        "--baseline-binary",
                        str(binary),
                    ]
                )
            self.assertEqual(arguments.model, model)
            self.assertEqual(arguments.package, package)

    def row(self, version, sample=0, latency=10):
        return {
            "version": version,
            "sample": sample,
            "context": 2048,
            "scenario": "cold",
            "prompt_sha256": "prompt",
            "response_sha256": "answer",
            "usage": {"completion_tokens": 1},
            "metrics": {"request_latency": {"ttft_ms": latency}},
        }

    def test_exact_prompt_lengths_unique_prefix_and_reproducibility(self):
        tokenizer = CharacterTokenizer()
        prompts = benchmark.prompts(tokenizer, [256, 2048], 2, "nonce")
        self.assertEqual(prompts, benchmark.prompts(tokenizer, [256, 2048], 2, "nonce"))
        for (_sample, context), prompt in prompts.items():
            self.assertEqual(
                len(
                    tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
                ),
                context,
            )
        self.assertNotEqual(prompts[0, 256][:32], prompts[1, 256][:32])

    def test_median_gate_keeps_raw_timing_separate_from_correctness(self):
        rows = [self.row("baseline"), self.row("candidate", latency=10.1)]
        self.assertTrue(benchmark.summarize(rows, 0.02)[0]["pass"])
        self.assertFalse(benchmark.summarize(rows, 0.005)[0]["pass"])

    def test_missing_duplicate_or_changed_transcript_fails(self):
        baseline, candidate = self.row("baseline"), self.row("candidate")
        for rows in ([baseline], [baseline, baseline, candidate]):
            with self.assertRaises(ValueError):
                benchmark.summarize(rows, 0.02)
        for field in ("response_sha256", "prompt_sha256"):
            changed = copy.deepcopy(candidate)
            changed[field] = "different"
            with self.assertRaisesRegex(ValueError, "transcript differs"):
                benchmark.summarize([baseline, changed], 0.02)

    def test_decode_uses_native_decode_time_per_token(self):
        rows = [self.row("baseline"), self.row("candidate")]
        for row in rows:
            row["scenario"] = "decode"
            row["native_delta"] = {"decode_wall_ms": 100, "decode_output_tokens": 50}
        summary = benchmark.summarize(rows, 0.02)[0]
        self.assertEqual(summary["metric"], "decode_ms_per_token")
        self.assertEqual(summary["baseline_median"], 2)


if __name__ == "__main__":
    unittest.main()
