import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tools import check_architecture


class ArchitectureTests(unittest.TestCase):
    def test_serving_modules_depend_on_lower_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "server"
            server.mkdir()
            (server / "server.py").write_text("from .frontend import Frontend")
            (server / "frontend.py").write_text("from .backend import Job")
            (server / "backend.py").write_text("from . import runtime")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for statement in (
                    "from .frontend import Frontend",
                    "from frontend import Frontend",
                    "from server.frontend import Frontend",
                    "from server import frontend",
                    "import server.frontend",
                ):
                    with self.subTest(statement=statement):
                        (server / "backend.py").write_text(statement)
                        self.assertEqual(
                            check_architecture.check(),
                            ["server/backend.py: imports upper serving layer frontend"],
                        )

    def test_serving_entrypoint_is_not_a_shared_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "server/output.py"
            source.parent.mkdir()
            source.write_text("def parse():\n    from .server import FrontendHandler\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    ["server/output.py: imports upper serving layer server"],
                )

    def test_workspace_policy_belongs_to_operators(self):
        symbols = (
            "PrefillAttentionWave",
            "prefillAttentionTiles",
            "kQ8VerifySplits",
            "kQ8PrefillAttentionTileRows",
            "moeMaximumTiles",
            "kMoePrefillTileRows",
            "kMoeDecodeTileRows",
            "Q4DecodeKind",
            "Q4DecodeShape",
            "Q4PrefillShape",
            "kQ4PrefillTileRows",
            "narrowAffineKind",
            "narrowResidualKind",
            "headKind",
            "gdnInputGroups",
            "attentionGroups",
            "addPrefill128",
            "LinearTile",
            "LinearConfig",
            "LinearSimdgroups",
            "PrefillSplitMultiplier",
            "PrefillAttentionConfig",
            "VerifySplitCount",
            "VerifyAttentionConfig",
            "AttentionScalePlacement",
            "MoeExpertTile",
            "MoeExpertSimdgroups",
            "MoeConfig",
            "DraftAttentionConfiguration",
            "selectorShards",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operator = root / "runtime/ops/Attention.cpp"
            operator.parent.mkdir(parents=True)
            operator.write_text("\n".join(symbols))
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for relative in (
                    "runtime/model/Runtime.cpp",
                    "runtime/engine/Scheduler.cpp",
                    "runtime/engine/Bootstrap.hpp",
                    "runtime/engine/Bootstrap.mm",
                    "runtime/engine/RuntimeResources.hpp",
                    "runtime/engine/RuntimeResources.mm",
                ):
                    source = root / relative
                    source.parent.mkdir(parents=True, exist_ok=True)
                    for symbol in symbols:
                        with self.subTest(source=relative, symbol=symbol):
                            source.write_text(f"auto size = ops::{symbol};")
                            errors = check_architecture.check()
                            layer = relative.split("/")[1]
                            self.assertEqual(
                                errors,
                                [
                                    f"{relative}: {layer} owns an operator workspace policy"
                                ],
                            )
                    source.unlink()

    def test_production_cannot_include_offline_tuning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime/ops/Linear.cpp"
            source.parent.mkdir(parents=True)
            source.write_text('#include "tuning/Measurement.hpp"\n')
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    [
                        "runtime/ops/Linear.cpp: production depends on "
                        "offline tuning tuning/Measurement.hpp"
                    ],
                )

    def test_plan_orchestration_and_storage_geometry_remain_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "runtime/model/Runtime.cpp"
            assembly = root / "runtime/engine/RuntimeResources.mm"
            model.parent.mkdir(parents=True)
            assembly.parent.mkdir(parents=True)
            model.write_text(
                "auto bytes = PagedAttention::prefillWorkspace();\n"
                "constexpr auto tileRows = kv::kPageTokens;\n"
                "auto rows = ExecutionLimits::draftQueryRows;\n"
                "auto rank = layout.selectorRank;\n"
            )
            assembly.write_text(
                '#include "model/Runtime.hpp"\n'
                "ops::ExecutionPlans plans(device);\n"
                "ops::OperatorChoices choices;\n"
                "plans.install(choices);\n"
                "ops::tuning::MeasurementOptions options;\n"
            )
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
