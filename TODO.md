# TODO List

### CenterNet

    Now that it runs cleanly, here’s the blunt truth: we've reached the “it works” phase. Next, we need to evolve it into the “it performs” phase. Here’s the ruthless next checklist:
    
    ##### ⚙️ Phase 1 – Model & Training Stability

    Loss balancing:
    Your heatmap loss will dominate. Tune coefficients:
    ```
    total_loss = 1.0 * hm_loss + 0.1 * wh_loss + 0.1 * off_loss
    ```

    Find the best ratios empirically.

    Use BCE focal loss for heatmap instead of MSE (CenterNet-style focal).
    The MSE makes positives too soft.

    Use sigmoid activation on heatmap output and clamp it to [1e-4, 1-1e-4].

    EMA model weights (Exponential Moving Average):
    Keeps inference smooth and prevents noisy training curves.

    ##### 🧠 Phase 2 – Data Improvements

    Better augmentations:
    ```
    A.MotionBlur(p=0.2)
    A.HueSaturationValue(p=0.3)
    A.RandomGamma(p=0.2)
    ```
    Avoid over-rotating cars (rotations >15° break geometry).

    Dataset balancing:
    If some classes dominate, use sampling weights.

    Auto-adjust Gaussian sigma based on box area:

    `sigma = torch.sqrt((w * h)) / 6`

    ##### 🚀 Phase 3 – Output Decoding

    Use top-K heatmap peaks instead of thresholding (torch.topk) for decoding.

    Add non-maximum suppression (NMS) on decoded boxes.

    When scaling boxes back to original image:
    ```
    boxes /= output_stride
    boxes *= (original_size / input_size)
    ```
    ##### 🧩 Phase 4 – Infrastructure

    Config-driven structure: YAML or JSON configs for model, dataset, and training.

    Checkpointing every N epochs with loss and val metrics.

    TensorBoard / WandB logging.

    Evaluation script: mAP calculation on validation set.

    ##### 🔥 Phase 5 – Performance

    Mixed precision (already done)

    Gradient clipping (to prevent explosion)

    CUDNN.benchmark = True

    Optionally export to ONNX for deployment.

    If you want, I can now rewrite your codebase into a structured “production-ready” training pipeline that includes:

    config.yaml

    train.py

    model.py

    dataset.py

    utils/ (losses, encoders, decoders, visualizations)

    inference.py
