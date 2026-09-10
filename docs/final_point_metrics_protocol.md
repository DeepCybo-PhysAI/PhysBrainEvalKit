# PhysBrain Final Point Localization Metrics Protocol

Protocol name: `physbrain_point_metrics_final_20260817`

Applicable benchmarks: Pixmo-Points, PointBench, Part-Affordance-2K,
RoboRefit, RoboSpatial, VABench-Point, PIOBench, Where2Place,
RefSpatial-Bench, and RoboAfford.

Implementation:

- `core/final_point_metrics.py`: protocol computation and aggregation.
- `core/framework_integration.py`: standard Qwen3-VL JSON parsing, coordinate
  conversion, mask-hit evaluation, and benchmark reference adapters.

## Per-sample evidence

Each point-localization sample produces four integer counts:

- `P = predicted_count`: number of parsed points; out-of-range points are still counted in P.
- `G = gt_count`: number of ground-truth targets.
- `M = matched_count`: number of ground-truth targets covered under the benchmark's matching rule.
- `H = raw_hit_count`: number of predicted points falling inside at least one acceptable ground-truth region.

M is used for recall; H is used for final non-strict precision. They must not be interchanged.

## Strict metrics

For an ordinary positive sample:

```text
TP = min(M, P, G)
FP = P - TP
FN = G - TP
```

Report micro-averaged metrics after aggregation. `strict_micro_f1` is retained as a
diagnostic strict metric; point-localization benchmark summaries use the non-strict
metric described below as their primary score.

Special rules for an explicit empty answer:

- PixMo no-object output with an explicit empty answer: `TP=1, FP=0, FN=0`.
- An explicit empty answer on a positive sample: `TP=0, FP=1, FN=G`.
- An unparseable answer with no predicted points: `TP=0, FP=0, FN=G`.

## Non-strict metrics

```text
precision = sum(H*) / sum(P)
recall = strict_recall
f1 = harmonic_mean(precision, recall)
```

Normally `H*=H`; for a PointBench counting mismatch, `H*=0`.
For point-localization benchmark summaries, the primary metric is
`non_strict_micro_f1`.

Each sample must store sufficient statistics independently:

- `non_strict_precision_hits`
- `non_strict_precision_predicted_points`
- `non_strict_recall_tp`
- `non_strict_recall_fn`

Non-strict precision and recall must not be represented as a single shared TP/FP/FN tuple.

## Benchmark-specific rules

### PointBench

For non-counting tasks, use the union mask: `M=min(mask_hits,G)` and `H=mask_hits`.
Counting tasks must enable the exact-count gate:

```text
if predicted_count != expected_count:
    TP = 0
    FP = predicted_count
    FN = expected_count
    non_strict_precision_hits = 0
```

Compute Overall with a unified micro aggregation over all 966 samples; do not aggregate only the counting subset. Three known mask-size anomaly IDs are `54`, `445`, and `775`. Use nearest-neighbor resize only, and assert that the anomaly set is unchanged for the complete dataset.

### PixMo

First perform Hungarian assignment between predicted points and ground-truth representative points using Euclidean distance. Then check whether each predicted point hits the mask of its assigned instance to obtain M. H is the number of predicted points that fall inside any instance mask. No-object samples use one sentinel ground-truth target.

### RoboSpatial

Use point metrics for 122 context samples and retain yes/no accuracy for 228 binary samples:

```text
strict_overall = (122 * context_strict_micro_f1 + binary_correct) / 350
non_strict_overall = (122 * context_non_strict_micro_f1 + binary_correct) / 350
```

RoboSpatial uses `non_strict_overall_score` as its primary benchmark score. The
strict mixed score remains available as `strict_overall_score` for diagnostics.

### RoboRefit

Use all 2,000 corrected masks. Bounding boxes do not contribute to scoring. Do not exclude the three samples whose prompt and ground-truth question text differ.

## Qwen3-VL output

Accept only the standard Qwen3-VL final JSON containing `point_2d`, an explicit `[]`/`No object`,
or an optional Markdown JSON fence. Do not scan reasoning text.

Coordinates are `[x,y]` in the 0--1000 range. Use floor and clamp the 1000 boundary:

```python
x_pixel = min(width - 1, int(x / 1000.0 * width))
y_pixel = min(height - 1, int(y / 1000.0 * height))
```

Out-of-range points are not converted to edge points, but remain included in P.

## Verification

```bash
python3 -m unittest \
  tests.test_final_point_metrics \
  tests.test_point_protocol \
  tests.test_point_benchmarks
```
