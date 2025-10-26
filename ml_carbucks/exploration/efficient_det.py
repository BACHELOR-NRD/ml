import torch
import datetime as dt
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
from effdet import create_model, create_loader
from effdet.data import resolve_input_config
from effdet.anchors import Anchors, AnchorLabeler

from ml_carbucks import DATA_CAR_DD_DIR, RESULTS_DIR
from ml_carbucks.utils.coco import CocoStatsEvaluator, create_dataset_custom
from ml_carbucks.utils.logger import setup_logger


logger = setup_logger("efficient_det")
logger.info("Starting efficient_det.py")

# CONFIGURATION

BATCH_SIZE = 8
IMG_SIZE = 320
NUM_CLASSES = 3
EPOCHS = 400
FREEZE_BACKBONE = False
LR = 5e-4
extra_args = dict(image_size=(IMG_SIZE, IMG_SIZE))
MODEL_NAME = "tf_efficientdet_d4"
RUNTIME = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
DATASET_LIMIT = 8

# TRAINING
logger.info(
    f"Model: {MODEL_NAME}, Image Size: {IMG_SIZE}, Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, Freeze Backbone: {FREEZE_BACKBONE}, LR: {LR}"
)
bench_train = create_model(
    model_name=MODEL_NAME,
    bench_task="train",
    num_classes=NUM_CLASSES,
    pretrained=True,
    redundant_bias=None,
    soft_nms=None,
    checkpoint_path="",
    checkpoint_ema=False,
    **extra_args,
)
model_train_config = bench_train.config
labeler = AnchorLabeler(
    Anchors.from_config(model_train_config),
    model_train_config.num_classes,
    match_threshold=0.2,
)

train_dataset = create_dataset_custom(
    name="train",
    img_dir=DATA_CAR_DD_DIR / "images" / "train",
    ann_file=DATA_CAR_DD_DIR / "instances_train.json",
    limit=DATASET_LIMIT,
)

train_input_config = resolve_input_config({}, model_train_config)
train_loader = create_loader(
    train_dataset,
    input_size=train_input_config["input_size"],
    batch_size=BATCH_SIZE,
    use_prefetcher=True,
    interpolation=train_input_config["interpolation"],
    mean=train_input_config["mean"],
    std=train_input_config["std"],
    num_workers=4,
    pin_mem=False,
    anchor_labeler=labeler,
)

val_dataset = create_dataset_custom(
    name="val",
    img_dir=DATA_CAR_DD_DIR / "images" / "val",
    ann_file=DATA_CAR_DD_DIR / "instances_val.json",
    limit=DATASET_LIMIT,
)

val_loader = create_loader(
    val_dataset,
    input_size=train_input_config["input_size"],
    batch_size=BATCH_SIZE,
    use_prefetcher=True,
    interpolation=train_input_config["interpolation"],
    mean=train_input_config["mean"],
    std=train_input_config["std"],
    num_workers=4,
    pin_mem=False,
    anchor_labeler=labeler,
)

train_evaluator = CocoStatsEvaluator(val_dataset, distributed=False, pred_yxyx=False)

if FREEZE_BACKBONE is True:
    for param in bench_train.model.backbone.parameters():  # type: ignore
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, bench_train.parameters()), lr=LR
    )
else:
    optimizer = torch.optim.AdamW(bench_train.parameters(), lr=LR, weight_decay=1e-5)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=5e-7
)
scheduler = None
bench_train = bench_train.cuda()
training_progress = defaultdict(list)


best_model_score = 0.0
best_model_weights = None
for epoch in range(EPOCHS):
    sll = sbl = scl = 0.0

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Check the dataset and annotations.")

    training_progress["epoch"].append(epoch + 1)
    training_progress["start_time"].append(
        dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    bench_train.train()
    batch_count = 0
    for input, target in tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} | Training batches"
    ):
        batch_count += 1
        output = bench_train(input, target)
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sll += round(loss.item(), 2)  # type: ignore
        sbl += round(output["box_loss"].item(), 2)  # type: ignore
        scl += round(output["class_loss"].item(), 2)  # type: ignore

    bench_train.eval()
    with torch.no_grad():
        for input, target in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} | Validation batches"
        ):
            output = bench_train(input, target)  # type: ignore
            train_evaluator.add_predictions(output["detections"], target)  # type: ignore

    stats = train_evaluator.evaluate()
    stats = [round(s, 4) for s in stats]
    train_evaluator.reset()

    if scheduler is not None:
        scheduler.step()

    training_progress["loss"].append(round(sll / batch_count, 4))
    training_progress["box_loss"].append(round(sbl / batch_count, 4))
    training_progress["class_loss"].append(round(scl / batch_count, 4))
    training_progress["val_mAP50-90"].append(stats[0])
    training_progress["val_mAP50"].append(stats[1])
    training_progress["val_mAR50-95"].append(stats[8])
    # NOTE: you could add more stats if needed, stats[0] is mAP 50-95
    if stats[0] > best_model_score:
        best_model_score = stats[0]
        best_model_weights = bench_train.model.state_dict()  # type: ignore
        best_model_save_path = RESULTS_DIR / f"best_{MODEL_NAME}_{RUNTIME}.pth"
        torch.save(best_model_weights, best_model_save_path)
        logger.info(
            f"New best model found at epoch {epoch + 1} with mAP50-90: {best_model_score}. Saved to {best_model_save_path}"
        )

    pd.DataFrame(training_progress).to_csv(
        f"training_{MODEL_NAME}_{RUNTIME}.csv", index=False
    )

    logger.info(
        f"Epoch {epoch + 1} completed. Loss: {training_progress['loss'][-1]}, Val mAP50-90: {training_progress['val_mAP50-90'][-1]}, Val mAP50: {training_progress['val_mAP50'][-1]}, Val mAR50-95: {training_progress['val_mAR50-95'][-1]}"
    )


train_state_dict = bench_train.model.state_dict()  # type: ignore

model_save_path = RESULTS_DIR / f"last_{MODEL_NAME}_{RUNTIME}.pth"
logger.info(f"Training completed. Saving model to {model_save_path}")
torch.save(train_state_dict, model_save_path)


# EVALUATION

logger.info("Starting evaluation phase")
bench_pred = create_model(
    model_name=MODEL_NAME,
    bench_task="predict",
    num_classes=NUM_CLASSES,
    pretrained=True,
    redundant_bias=None,
    soft_nms=None,
    checkpoint_path="",
    checkpoint_ema=False,
    **extra_args,
)
best_trained_path = RESULTS_DIR / f"best_{MODEL_NAME}_{RUNTIME}.pth"
best_trained_state_dict = torch.load(best_trained_path)
bench_pred.model.load_state_dict(best_trained_state_dict)  # type: ignore
model_pred_config = bench_pred.config
bench_pred = bench_pred.cuda()

test_dataset = create_dataset_custom(
    name="test",
    img_dir=DATA_CAR_DD_DIR / "images" / "test",
    ann_file=DATA_CAR_DD_DIR / "instances_test.json",
)

test_input_config = resolve_input_config({}, model_pred_config)
test_loader = create_loader(
    test_dataset,
    input_size=test_input_config["input_size"],
    batch_size=BATCH_SIZE,
    use_prefetcher=True,
    interpolation=test_input_config["interpolation"],
    mean=test_input_config["mean"],
    std=test_input_config["std"],
    num_workers=4,
    pin_mem=False,
)
logger.info(f"Len test loader: {len(test_loader)}")
evaluator = CocoStatsEvaluator(test_dataset, distributed=False, pred_yxyx=False)
bench_pred.eval()


with torch.no_grad():
    for input, target in tqdm(test_loader):
        output = bench_pred(input, img_info=target)
        evaluator.add_predictions(output, target)

stats = evaluator.evaluate()
stats = [round(s, 4) for s in stats]
logger.info(
    f"Test set evaluation stats: mAP50-90: {stats[0]}, mAP50: {stats[1]}, mAR50-95: {stats[8]}"
)
