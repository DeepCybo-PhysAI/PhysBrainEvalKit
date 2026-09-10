import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.export_benchmark_result_bundle import export_bundle


class ResultBundleTests(unittest.TestCase):
    def test_exports_resume_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_json = root / "logs" / "results.json"
            output_dir = root / "output" / "ERQA"
            result_json.parent.mkdir(parents=True)
            result_json.write_text(
                json.dumps(
                    {
                        "results": [{"idx": 0, "is_correct": True}],
                        "statistics": {
                            "overall_accuracy": 1.0,
                            "total_samples": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            export_bundle(
                Namespace(
                    benchmark="ERQA",
                    output_dir=output_dir,
                    result_json=result_json,
                    log_file=output_dir / "run.log",
                    model_name="qwen3-vl-public",
                    model_path=root / "model",
                    entry_script="eval_erqa.py",
                    dataset="FlagEval/ERQA",
                    split="test",
                    start_time="2026-09-10T00:00:00+00:00",
                    end_time="2026-09-10T00:01:00+00:00",
                    command="eval_erqa.py --backend hf",
                    inference_params_json='{"backend":"hf","temperature":0.0}',
                )
            )

            self.assertEqual(
                json.loads((output_dir / "all_samples.json").read_text()),
                [{"idx": 0, "is_correct": True}],
            )
            metadata = json.loads((output_dir / "meta_result.json").read_text())
            self.assertEqual(metadata["benchmark"], "ERQA")
            self.assertEqual(metadata["num_samples"], 1)
            self.assertEqual(metadata["metrics"]["overall_accuracy"], 1.0)
            self.assertEqual(metadata["inference_params"]["backend"], "hf")


if __name__ == "__main__":
    unittest.main()
