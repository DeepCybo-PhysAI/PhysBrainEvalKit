import ast
import unittest
from pathlib import Path

from scripts.benchmark_registry import common_cli


ROOT = Path(__file__).resolve().parents[1]


def parser_defaults(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        option = node.args[0].value
        if not isinstance(option, str) or not option.startswith("--"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            try:
                defaults[option] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                continue
    return defaults


class GenerationDefaultsTests(unittest.TestCase):
    def test_runner_common_cli_is_greedy_and_reproducible(self):
        argv = common_cli("model", Path("/tmp/model"), "qwen3")
        values = dict(zip(argv[::2], argv[1::2]))
        self.assertEqual(values["--temperature"], "0.0")
        self.assertEqual(values["--top_p"], "1.0")
        self.assertEqual(values["--top_k"], "-1")
        self.assertEqual(values["--repetition_penalty"], "1.05")
        self.assertEqual(values["--presence_penalty"], "0.0")
        self.assertEqual(values["--max_tokens"], "1024")

    def test_direct_entry_points_use_the_same_sampling_defaults(self):
        for path in sorted(ROOT.glob("eval_*.py")):
            defaults = parser_defaults(path)
            if "--temperature" not in defaults:
                continue
            if path.name == "eval_mmsi_bench.py":
                self.assertEqual(
                    (
                        defaults["--temperature"],
                        defaults["--top_p"],
                        defaults["--top_k"],
                    ),
                    (0.7, 0.8, 20),
                )
                continue
            self.assertEqual(defaults["--temperature"], 0.0, path.name)
            self.assertEqual(defaults["--top_p"], 1.0, path.name)
            self.assertEqual(defaults["--top_k"], -1, path.name)
            self.assertEqual(defaults["--seed"], 3407, path.name)

    def test_public_plan_max_tokens_match_registry(self):
        expected = {
            "eval_erqa.py": 1024,
            "eval_robospatial.py": 4096,
            "eval_egoplan2.py": 1024,
            "eval_sat.py": 1024,
            "eval_where2place.py": 4096,
            "eval_refspatial.py": 1024,
            "eval_partafford.py": 4096,
            "eval_sharerobot_v.py": 1024,
            "eval_vabench_v.py": 1024,
            "eval_qspatial.py": 1024,
            "eval_vabench_point.py": 4096,
            "eval_pixmo_points.py": 4096,
            "eval_roboafford.py": 1024,
            "eval_pio.py": 4096,
            "eval_roborefit.py": 1024,
            "eval_blink.py": 1024,
            "eval_cvbench.py": 1024,
            "eval_vsi_bench.py": 1024,
            "eval_embspatial.py": 1024,
            "eval_pointbench.py": 4096,
            "eval_cosmos.py": 1024,
            "eval_robovqa.py": 128,
            "eval_vlabench.py": 512,
            "eval_erqa_plus.py": 128,
            "eval_3dsrbench.py": 128,
            "eval_viewspatial.py": 128,
            "eval_mindcube.py": 128,
            "eval_mmsi_bench.py": 128,
        }
        for filename, expected_value in expected.items():
            defaults = parser_defaults(ROOT / filename)
            self.assertEqual(defaults["--max_tokens"], expected_value, filename)


if __name__ == "__main__":
    unittest.main()
