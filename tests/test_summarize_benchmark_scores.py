import unittest

from scripts.summarize_benchmark_scores import pick_primary_metric


class SummarizeBenchmarkScoresTests(unittest.TestCase):
    def test_refspatial_uses_non_strict_micro_f1(self):
        metric_name, score = pick_primary_metric(
            "RefSpatial-Bench",
            {
                "strict_micro_f1": 0.25,
                "non_strict_micro_f1": 0.75,
            },
        )

        self.assertEqual(metric_name, "non_strict_micro_f1")
        self.assertEqual(score, 0.75)

    def test_robospatial_uses_non_strict_overall_score(self):
        metric_name, score = pick_primary_metric(
            "RoboSpatial",
            {
                "overall_accuracy": 0.30,
                "strict_overall_score": 0.40,
                "non_strict_overall_score": 0.80,
                "non_strict_micro_f1": 0.70,
            },
        )

        self.assertEqual(metric_name, "non_strict_overall_score")
        self.assertEqual(score, 0.80)

    def test_point_benchmark_uses_non_strict_micro_f1(self):
        metric_name, score = pick_primary_metric(
            "RoboRefit",
            {
                "strict_micro_f1": 0.20,
                "non_strict_micro_f1": 0.90,
                "overall_accuracy": 0.20,
            },
        )

        self.assertEqual(metric_name, "non_strict_micro_f1")
        self.assertEqual(score, 0.90)


if __name__ == "__main__":
    unittest.main()
