import torch
import datetime as dt

import cloudpickle as cpkl
from effdet import create_model, create_loader, create_evaluator
from effdet.data import resolve_input_config
from effdet.anchors import Anchors, AnchorLabeler

from ml_carbucks import DATA_CAR_DD_DIR
from ml_carbucks.utils import create_dataset_custom

# CONFIGURATION

BATCH_SIZE = 16
IMG_SIZE = 320
NUM_CLASSES = None
EPOCHS = 150
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

if FREEZE_BACKBONE is True:
    for param in bench_train.model.backbone.parameters():  # type: ignore
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, bench_train.parameters()), lr=LR
    )
else:
    optimizer = torch.optim.AdamW(bench_train.parameters(), lr=LR)

bench_train = bench_train.cuda()
bench_train.train()
for epoch in range(EPOCHS):
    sll = 0.0
    sbl = 0.0
    scl = 0.0
    for batch_idx, (input, target) in enumerate(train_loader):
        output = bench_train(input, target)
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sll += loss.item()  # type: ignore
        sbl += output["box_loss"].item()  # type: ignore
        scl += output["class_loss"].item()  # type: ignore

    sll /= len(train_loader)
    sbl /= len(train_loader)
    scl /= len(train_loader)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {sll}, Box Loss: {sbl}, Class Loss: {scl}")  # type: ignore
    # NOTE: it could be nice to add validation here and later just test the best model insetad of evalaution as it is now


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

val_dataset = create_dataset_custom(
    name="val",
    img_dir=DATA_CAR_DD_DIR / "images" / "val",
    ann_file=DATA_CAR_DD_DIR / "instances_val.json",
)

input_config = resolve_input_config({}, model_pred_config)
loader = create_loader(
    val_dataset,
    input_size=input_config["input_size"],
    batch_size=BATCH_SIZE,
    use_prefetcher=True,
    interpolation=input_config["interpolation"],
    mean=input_config["mean"],
    std=input_config["std"],
    num_workers=4,
    pin_mem=False,
)
evaluator = create_evaluator("coco", val_dataset, pred_yxyx=False)
bench_pred.eval()


with torch.no_grad():
    for i, (input, target) in enumerate(loader):
        output = bench_pred(input, img_info=target)
        evaluator.add_predictions(output, target)

        if i % 10 == 0:
            print(f"Eval {i}/{len(loader)}")

metrics = evaluator.evaluate()
print(metrics)
cpkl.dump(metrics, open(f"{MODEL_NAME}_{RUNTIME}_metrics.cpkl", "wb"))
