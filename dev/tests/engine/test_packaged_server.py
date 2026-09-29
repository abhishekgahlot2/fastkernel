import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.engine.test_documents import document_block
from dev.tools import package


class PackagedServerTests(unittest.TestCase):
    def test_staged_server_imports_and_renders_without_the_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            stage = Path(directory) / "stage"
            stage.mkdir()
            for folder, names in (
                ("server", package.SERVER_FILES),
                ("install", package.INSTALL_FILES),
                ("install/completions", package.COMPLETION_FILES),
            ):
                (root / folder).mkdir(parents=True)
                for name in names:
                    shutil.copy2(package.ROOT / folder / name, root / folder / name)
            (root / "build").mkdir()
            for name in ("splash", "splash.metallib"):
                (root / "build" / name).write_bytes(b"unused CPU test fixture")
            for name in ("LICENSE",):
                shutil.copy2(package.ROOT / name, root / name)
            with mock.patch.object(package, "ROOT", root):
                package.stage_runtime(stage, "test")
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import json, os, sys; sys.path.insert(0, os.getcwd()); "
                    "from server import server, documents; "
                    "print(documents.document_content(json.load(sys.stdin))[0]['text'])",
                ],
                cwd=stage,
                input=json.dumps(document_block()),
                text=True,
                capture_output=True,
                timeout=40,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ALPHA 42", result.stdout)


if __name__ == "__main__":
    unittest.main()
