import datetime as dt
from typing import Any

from tqdm import tqdm
from effdet import (  # noqa F401
    create_model,
    unwrap_bench,
    create_loader,
    create_dataset,
    create_evaluator,
)
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from timm.models.layers import set_layer_config  # type: ignore # noqa F401
from effdet.data import resolve_input_config, SkipSubset  # noqa F401
from effdet.anchors import Anchors, AnchorLabeler
import torch  # noqa F401

from ml_carbucks.utils.effdet_extension import create_dataset_custom
from ml_carbucks.utils.logger import setup_logger
from ml_carbucks.utils.result_saver import ResultSaver  # noqa F401
from ml_carbucks import RESULTS_DIR, DATA_DIR


logger = setup_logger("effdet_v2", log_file="/home/bachelor/ml-carbucks/logs/logs.log")


class Args:
    def __init__(self, **entries):
        for key, value in entries.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        return self.__dict__.get(name, None)

    def vars(self):
        return self.__dict__


def create_datasets_and_loaders(
    args: Args,
    model_config: Any,
    transform_train_fn=None,
    transform_eval_fn=None,
    collate_fn=None,
):

    input_config = resolve_input_config(args, model_config)

    dataset_train = create_dataset_custom(
        img_dir=DATA_DIR / "car_dd" / "images" / "train",
        ann_file=DATA_DIR / "car_dd" / "instances_train_curated.json",
        has_labels=True,
    )

    dataset_eval = create_dataset_custom(
        img_dir=DATA_DIR / "car_dd" / "images" / "val",
        ann_file=DATA_DIR / "car_dd" / "instances_val_curated.json",
        has_labels=True,
    )

    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config),
            model_config.num_classes,
            match_threshold=0.5,
        )

    loader_train = create_loader(
        dataset_train,
        input_size=input_config["input_size"],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        # color_jitter=args.color_jitter,
        # auto_augment=args.aa,
        interpolation=input_config["interpolation"],
        fill_color=input_config["fill_color"],
        mean=input_config["mean"],
        std=input_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_train_fn,
        collate_fn=collate_fn,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=input_config["input_size"],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=input_config["interpolation"],
        fill_color=input_config["fill_color"],
        mean=input_config["mean"],
        std=input_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
    )

    evaluator = create_evaluator(
        args.dataset, loader_eval.dataset, distributed=args.distributed, pred_yxyx=False
    )

    return loader_train, loader_eval, evaluator


# NOTE: EfficientDet uses YXYX format and 0 class is reserved for background so own labels from 0..N-1 are incorrect.
def main():

    args = Args(
        model="tf_efficientdet_d0",
        pretrained_backbone=False,
        pretrained=True,
        prefetcher=True,
        device="cuda",
        amp=False,
        num_classes=3,
        opt="momentum",
        weight_decay=1e-5,
        lr=0.008,
        epochs=50,
        dataset="coco",
        root="/home/maindamian/efficientdet-pytorch/car_dd",
        batch_size=8,
        bench_labeler=False,
        distributed=False,
        pin_mem=False,
        workers=4,
        reprob=0.0,
        remode="pixel",
        recount=1,
        runtime_stamp=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        torchscript=False,
        initial_checkpoint="",
        redundant_bias=None,
        train_interpolation="random",
    )

    bench_train = create_model(
        args.model,
        bench_task="train",
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        redundant_bias=args.redundant_bias,
        label_smoothing=args.smoothing,
        legacy_focal=args.legacy_focal,
        jit_loss=args.jit_loss,
        soft_nms=args.soft_nms,
        bench_labeler=args.bench_labeler,
        checkpoint_path=args.initial_checkpoint,
    )

    bench_train_config = bench_train.config
    bench_train.cuda()
    logger.info("here")
    optimizer = create_optimizer_v2(
        bench_train,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lr_scheduler, num_epochs = create_scheduler_v2(optimizer)

    loader_train, loader_eval, evaluator = create_datasets_and_loaders(
        args, bench_train_config
    )

    # parser_max_label = loader_train.dataset.parser.max_label  # type: ignore
    # config_num_classes = bench_train_config.num_classes

    # if parser_max_label != config_num_classes:
    #     logger.error(
    #         f"Number of classes in dataset ({parser_max_label}) does not match "
    #         f"model config ({config_num_classes})."
    #     )
    #     exit(1)

    saver = ResultSaver(
        path=RESULTS_DIR / "effdet_v2",
        name=f"{args.model}_{args.runtime_stamp}",
    )

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs} starting...")

        # Training and evaluation logic would go here
        bench_train.train()

        total_loss = 0.0
        for inputs, targets in tqdm(loader_train):
            output = bench_train(inputs, targets)
            loss = output["loss"]
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        bench_train.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in loader_eval:
                val_output = bench_train(val_inputs, val_targets)
                val_loss = val_output["loss"]
                val_total_loss += val_loss.item()
                evaluator.add_predictions(val_output["detections"], val_targets)

                torch.cuda.synchronize()
        val_map = evaluator.evaluate().item()  # type: ignore
        evaluator.reset()
        saver.save(
            epoch=epoch + 1, loss=total_loss, val_map=val_map, val_loss=val_total_loss
        ).plot(show=False)

        logger.info(f"Epoch {epoch + 1}/{args.epochs} completed.")


if __name__ == "__main__":
    main()
