import json
import torch
import datetime as dt

import pandas as pd
from effdet import create_model, create_loader, create_evaluator
from effdet.data import resolve_input_config
from effdet.anchors import Anchors, AnchorLabeler

from ml_carbucks import DATA_CAR_DD_DIR
from ml_carbucks.utils.coco import CocoStatsEvaluator, create_dataset_custom

# CONFIGURATION

BATCH_SIZE = 16
IMG_SIZE = 320
NUM_CLASSES = None
EPOCHS = 50
FREEZE_BACKBONE = False
LR = 1e-4
extra_args = dict(image_size=(IMG_SIZE, IMG_SIZE))
MODEL_NAME = "tf_efficientdet_d4"
RUNTIME = dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# TRAINING

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
    match_threshold=0.5,
)

train_dataset = create_dataset_custom(
    name="train",
    img_dir=DATA_CAR_DD_DIR / "images" / "train",
    ann_file=DATA_CAR_DD_DIR / "instances_train.json",
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
    optimizer = torch.optim.AdamW(bench_train.parameters(), lr=LR)

bench_train = bench_train.cuda()
training_progress = {
    "epoch": [],
    "loss": [],
    "box_loss": [],
    "class_loss": [],
    "val_map50-90": [],
    "start_time": [],
}

for epoch in range(EPOCHS):
    sll = 0.0
    sbl = 0.0
    scl = 0.0
    stats = [-1]
    bench_train.train()
    DISPLAY_STATS_ON_EPOCH = (epoch + 1) % 10 == 0 or epoch == 0
    training_progress["start_time"].append(
        dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Check the dataset and annotations.")

    for batch_idx, (input, target) in enumerate(train_loader):
        output = bench_train(input, target)
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sll += loss.item()  # type: ignore
        sbl += output["box_loss"].item()  # type: ignore
        scl += output["class_loss"].item()  # type: ignore

        if DISPLAY_STATS_ON_EPOCH:
            bench_train.eval()
            with torch.no_grad():
                for i, (input, target) in enumerate(val_loader):
                    output = bench_train(input, target)  # type: ignore
                    train_evaluator.add_predictions(output["detections"], target)  # type: ignore

            stats = train_evaluator.evaluate()
            train_evaluator.reset()

    all = round(sll / len(train_loader), 4)
    abl = round(sbl / len(train_loader), 4)
    acl = round(scl / len(train_loader), 4)
    training_progress["epoch"].append(epoch + 1)
    training_progress["loss"].append(all)
    training_progress["box_loss"].append(abl)
    training_progress["class_loss"].append(acl)
    training_progress["val_map50-90"].append(stats[0])
    # NOTE: you could add more stats if needed, stats[0] is mAP 50-95

    pd.DataFrame(training_progress).to_csv(
        f"training_{MODEL_NAME}_{RUNTIME}.csv", index=False
    )
    if DISPLAY_STATS_ON_EPOCH:
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Loss: {all}, BoxLoss: {abl}, ClassLoss: {acl}, val_map50-90: {stats[0]}"
        )


train_state_dict = bench_train.model.state_dict()  # type: ignore

torch.save(train_state_dict, f"{MODEL_NAME}_{RUNTIME}.pth")


# EVALUATION

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
bench_pred.model.load_state_dict(train_state_dict)  # type: ignore
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
evaluator = create_evaluator("coco", test_dataset, pred_yxyx=False)
bench_pred.eval()


with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        output = bench_pred(input, img_info=target)
        evaluator.add_predictions(output, target)

        if i % 10 == 0:
            print(f"Eval {i}/{len(test_loader)}")

metrics = evaluator.evaluate()
print(metrics)

with open(f"eval_{MODEL_NAME}_{RUNTIME}.json", "w") as f:
    json.dump({"metrics": metrics}, f)
