# Carbucks-Only Adapter Runs

## Dataset References
- Definitions for `carbucks_*` splits (raw, cleaned, balanced) live in `milestones/m3_carbucks_data_analysis_and_effdet_loaders.ipynb:77-99`.
- EfficientDet loader experiments covering the same splits are documented in `milestones/m3_carbucks_data_analysis_and_effdet_loaders.ipynb:1336-1477`.

## Ultralytics (YOLO & RT-DETR)
- **`carbucks_balanced_yolo2`**
  - Data: `data/carbucks_balanced`
  - Config: `results/yolo_runs/carbucks_balanced_yolo2/args.yaml`
  - Best mAP50 / mAP50-95: `0.5768 / 0.3358` (epoch 150)
  - Notes: YOLO11-L @ 1024 px, strongest Carbucks-only checkpoint.
- **`rtdetr_carbucks_experiment_large`**
  - Data: `data/carbucks`
  - Config: `results/debug/carbucks_ultralytics/rtdetr_carbucks_experiment_large/args.yaml`
  - Best mAP50 / mAP50-95: `0.1427 / 0.0642` (epochs 115 / 57)
  - Notes: RT-DETR-L @ 640 px, long run with plateau after ~120 epochs.
- **`yolo_carbucks_experiment_large`**
  - Data: `data/carbucks`
  - Config: `results/debug/carbucks_ultralytics/yolo_carbucks_experiment_large/args.yaml`
  - Best mAP50 / mAP50-95: `0.1267 / 0.0515` (epochs 157 / 199)
  - Notes: YOLO11-L @ 640 px, extended schedule.
- **`yolo11l_carbucks_balanced_test_long`**
  - Data: `data/carbucks_balanced`
  - Config: `results/debug/carbucks_ultralytics/yolo11l_carbucks_balanced_test_long/args.yaml`
  - Best mAP50 / mAP50-95: `0.1053 / 0.0386` (epoch 99)
  - Notes: Baseline 100-epoch run.
- **`yolo11l_carbucks_balanced_test_long_lr_big`**
  - Data: `data/carbucks_balanced`
  - Config: `results/debug/carbucks_ultralytics/yolo11l_carbucks_balanced_test_long_lr_big/args.yaml`
  - Best mAP50 / mAP50-95: `0.0304 / 0.0087` (epochs 18 / 17)
  - Notes: Large-LR experiment that regressed quickly.
- **`yolo11m_carbucks_cleaned_test3`**
  - Data: `data/carbucks_cleaned`
  - Config: `results/debug/carbucks_ultralytics/yolo11m_carbucks_cleaned_test3/args.yaml`
  - Best mAP50 / mAP50-95: `0.0733 / 0.0281` (epoch 20)
  - Notes: Medium model on cleaned split.
- **`yolo11m_carbucks_test`**
  - Data: `data/carbucks`
  - Config: `results/debug/carbucks_ultralytics/yolo11m_carbucks_test/args.yaml`
  - Best mAP50 / mAP50-95: `0.0539 / 0.0209` (epoch 20)
  - Notes: Medium model on raw split.
- **`yolo11m_carbucks_balanced_test`**
  - Data: `data/carbucks_balanced`
  - Config: `results/debug/carbucks_ultralytics/yolo11m_carbucks_balanced_test/args.yaml`
  - Best mAP50 / mAP50-95: `0.0452 / 0.0168` (epoch 20)
  - Notes: Short balanced run.

_Folders `yolo11m_carbucks_balanced_test2/3/4` currently only contain weights and visualization images, so no metrics are logged yet._

## EfficientDet Adapter Runs
- **`effdet_carbucks_fixed_exif.csv`**
  - Data: `data/carbucks` (EXIF corrected)
  - Log: `results/debug/carbucks/effdet_carbucks_fixed_exif.csv`
  - Best mAP50-95: `0.0241` (epoch 10)
  - Notes: Same run as baseline but after orientation fixes.
- **`effdet_carbucks.csv`**
  - Data: `data/carbucks`
  - Log: `results/debug/carbucks/effdet_carbucks.csv`
  - Best mAP50 / mAP50-95: `0.0258 / 0.0078` (epoch 10)
  - Notes: Before EXIF cleanup.
- **`carbucks_train_raw.csv`**
  - Data: `data/carbucks`
  - Log: `results/debug/effdet_datasets_experiments/carbucks_train_raw.csv`
  - Best mAP50 / mAP50-95: `0.0946 / 0.0337` (epoch 8)
  - Notes: Custom loader sweep.
- **`carbucks_train_clean.csv`**
  - Data: `data/carbucks_cleaned`
  - Log: `results/debug/effdet_datasets_experiments/carbucks_train_clean.csv`
  - Best mAP50 / mAP50-95: `0.0845 / 0.0300` (epoch 10)
  - Notes: Cleaned split results.
- **`carbucks_train_raw_original.csv`**
  - Data: `data/carbucks`
  - Log: `results/debug/effdet_datasets_experiments/carbucks_train_raw_original.csv`
  - Best mAP50 / mAP50-95: `0.0875 / 0.0320` (epoch 8 / 7)
  - Notes: Using the original loader.
- **`carbucks_train_clean_original.csv`**
  - Data: `data/carbucks_cleaned`
  - Log: `results/debug/effdet_datasets_experiments/carbucks_train_clean_original.csv`
  - Best mAP50 / mAP50-95: `0.0753 / 0.0275` (epoch 9)
  - Notes: Original loader + cleaned split.
- **`effdet_extended_carbucks.csv`**
  - Data: `data/carbucks`
  - Log: `results/debug/efficientdet_datasets/effdet_extended_carbucks.csv`
  - Best mAP50 / mAP50-95: `0.1491 / 0.0631` (epochs 28 / 29)
  - Notes: Long EfficientDet run; best non-Ultralytics metrics to date.

## Next Steps
1. Promote `carbucks_balanced_yolo2` weights into ensemble configs to align evals with Carbucks-focused training.
2. Re-run Faster R-CNN / EfficientDet on the balanced split using the loader setup proven above to close the gap with YOLO results.
