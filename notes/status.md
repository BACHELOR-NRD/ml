# Ensemble Snapshot (ml_carbucks)

## Adapters in Plain Words
- An **adapter** is a thin wrapper that makes any detector (YOLO, RT-DETR, Faster R-CNN, EfficientDet, etc.) look the same from the outside.
- Each adapter follows the `BaseDetectionAdapter` contract: it must know how to `setup`, `fit`, `evaluate`, `predict`, and `save`.
- Because the interface is shared, we can swap models or drop them into an ensemble without rewriting training or evaluation code.
- Adapters also keep device handling, preprocessing, and prediction formatting consistent, so downstream tools only read one data shape.

## Why Build an Ensemble?
- The shared adapter interface lets `EnsembleModel` spin up many detectors with the same dataset call.
- For every image, the ensemble gathers each adapter’s boxes and then fuses them with simple stacking, Non-Maximum Suppression (NMS), or Weighted Boxes Fusion (WBF).
- Before merging, `normalize_scores()` can scale or weight confidences so a noisy model does not drown out the rest.
- Mixing models that make different mistakes (missed scratches, over-detected reflections, etc.) boosts recall and keeps precision under control.

## Who Is in the Ensemble?
| Adapter | Backbone & Framework | Why It’s Useful | Watch-outs |
| --- | --- | --- | --- |
| `YoloUltralyticsAdapter` | YOLO11-L (Ultralytics) | Fast single-stage conv net with strong built-in augmentations for tiny defects. | Already does its own NMS, so duplicate boxes may disappear unless thresholds are tuned. |
| `RtdetrUltralyticsAdapter` | RT-DETR transformer | Transformer reasoning handles cluttered scenes and elongated dents; outputs dense candidate boxes. | Needs extra NMS/score cleanup because Ultralytics skips it for RT-DETR. |
| `FasterRcnnAdapter` | ResNet50-FPN two-stage (Torchvision) | Region proposals give calibrated scores and crisp localization. | Slower inference and requires 1-based class IDs because background uses 0. |
| `EfficientDetAdapter` | EfficientDet-D0 (timm/effdet) | BiFPN balances speed/accuracy and supports both EffDet’s loader and our Albumentations loader. | Sensitive to how images are preprocessed; mixing loaders can shift scales. |

### How They Fit Together
- **Different brains:** YOLO is convolution-heavy, RT-DETR is transformer-based, Faster R-CNN is two-stage, and EfficientDet mixes both ideas. Variety = broader coverage.
- **Different augmentations:** Ultralytics stacks offer heavy augmentations, while Faster R-CNN/EfficientDet can run in “clean” mode—this gives the ensemble multiple data views.
- **Different suppression habits:** RT-DETR keeps overlaps, YOLO prunes aggressively, so WBF benefits from their contrasting behaviours.

## How the Ensemble Runs
1. **Setup once:** `EnsembleModel(classes, adapters).setup()` calls each adapter’s `setup` so weights land on the right device.
2. **Get predictions:** For every batch from `create_clean_loader`, the ensemble calls `.predict()` on each adapter, which always returns tensors with the same schema.
3. **Fuse results:** `fuse_adapters_predictions()` merges the per-image lists via:
   - plain stacking with a confidence cut,
   - NMS for fast duplicate removal,
   - or WBF to average overlapping boxes instead of dropping them.
4. **Score the mix:** `torchmetrics.MeanAveragePrecision` computes mAP on the fused predictions, so comparisons with single models stay consistent.

## What We Measure
- **Primary metric – mAP@0.50:0.95:** Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 in 0.05 steps, matching COCO evaluation. A higher value means the detector stays accurate even when we require very tight box overlap.
- **Quick sanity check – mAP@0.50:** Average precision at a single IoU threshold of 0.50. It is more forgiving and shows whether the model is at least catching the objects roughly in the right place.
- **Class awareness:** Each adapter reports metrics per class ID so we can see, for example, if “scratch” lags behind “dent.” The ensemble uses the same schema, so improvements or regressions per damage type are easy to track.
- **Supporting signals:** During fusion experiments we also monitor per-image detection counts and confidence distributions to confirm that score normalisation or trust weights behave as expected, but mAP drives go/no-go decisions.

## Data & Training Summary
- **Dataset shrinkage:** The original public release had ~2.8k train / 0.8k eval images, whereas CarBucks now holds ~1.9k train / 0.3k eval, so every split is leaner.
- **Tougher visuals:** Our captures are less “perfect”—lighting shifts, odd camera angles, varying object scales—which makes the detectors work harder than on the curated public shots.
- **Early plateaus:** Loss curves fall but mAP stagnates quickly, pointing to either overfitting (memorising the smaller train set) or underfitting (model capacity/learning schedule mismatch).
- **Class imbalance & empties:** About 12 % of the photos had no annotations; pruning them helped but class distribution remains skewed. Down-weighting the majority (scratches) dropped its mAP while other classes stayed roughly the same, so balancing strategies need more controlled experiments.

## Quick Takeaways
- Adapters keep the codebase simple: one training/evaluation path works for every backbone.
- The ensemble gains robustness because each detector spots different damage patterns.
- Score normalisation and trust weights stop one overconfident model from taking over.
- Swapping or adding detectors is low-effort—build the adapter once, and the ensemble picks it up automatically.
