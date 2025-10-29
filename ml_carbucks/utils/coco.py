import os
import json
import datetime as dt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, cast

import torch
import torch.distributed as dist
from effdet.data.parsers import CocoParserCfg, create_parser
from effdet.data.dataset import DetectionDatset
from effdet.evaluator import Evaluator
from pycocotools.cocoeval import COCOeval


class FilteredDataset:
    """A dataset wrapper that only exposes a subset of the base dataset.
    This is useful to limit the number of images used during training or evaluation."""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]

    # delegate attribute access for attributes not found on this wrapper
    def __getattr__(self, name):
        return getattr(self.base_dataset, name)

    # Explicitly expose parser and transform properties with setter delegation
    @property
    def parser(self):
        return self.base_dataset.parser

    @property
    def transform(self):
        return self.base_dataset.transform

    @transform.setter
    def transform(self, t):
        # When create_loader sets dataset.transform = ..., delegate to base dataset
        self.base_dataset.transform = t


def create_dataset_custom(
    img_dir: Path | str,
    ann_file: Path | str,
    has_labels: bool,
    limit: Optional[int] = None,
    limit_mode: str = "first",
    seed: int | None = None,
):
    """
    Create a COCO dataset with optional limiting of the number of images.

    Gives more flexibility than the standard effdet dataset creation as it
    allows for specifying img_dir and ann_file separately.

    Args:
        name: Name of the dataset.
        img_dir: Directory containing the images.
        ann_file: Path to the COCO annotation file.
        has_labels: Whether the dataset has labels (True for train/val, False for predictions).
        limit: If positive, limits the dataset to this many images.
        limit_mode: 'first' to take the first `limit` images, 'random'
                    to take a random sample of `limit` images.
        seed: Random seed to use if limit_mode is 'random'.
    Returns:
        The created dataset or a list of datasets if multiple are created.
    """

    # NOTE: This parser configuration needs YXYX annotation format.
    # This is because of the insane way the EfficientDet model was written.
    # It just is what it is.
    parser_cfg = CocoParserCfg(
        ann_filename=ann_file,  # type: ignore
        has_labels=has_labels,
    )
    dataset = DetectionDatset(
        data_dir=img_dir,  # type: ignore
        parser=create_parser("coco", cfg=parser_cfg),
    )
    # NOTE: this is a fix for missing 'info' field in some COCO annotations
    if "info" not in dataset.parser.coco.dataset:  # type: ignore
        dataset.parser.coco.dataset["info"] = {  # type: ignore
            "description": "unknown",
            "url": "unknown",
            "version": "0.1",
            "year": dt.datetime.now().year,
            "contributor": "unknown",
            "date_created": "unknown",
        }
    # If limit is set and positive, create a FilteredDataset so the loader
    # only iterates over up to `limit` items. This limits images inside the dataset.
    if limit is not None and int(limit) > 0:
        n = min(int(limit), len(dataset))
        if limit_mode == "random":
            import random

            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), n)
        else:
            # default 'first' behaviour
            indices = list(range(n))
        dataset = FilteredDataset(dataset, indices)

    return dataset


class CocoStatsEvaluator(Evaluator):
    """
    Similar to CocoEvaluator from effdet library but returns ALL COCO metrics, not just mAP50-95.
    """

    def __init__(self, dataset, distributed=False, pred_yxyx=False):
        super().__init__(distributed=distributed, pred_yxyx=pred_yxyx)
        self._dataset = dataset.parser
        self.coco_api = dataset.parser.coco

    def reset(self):
        self.img_indices = []
        self.predictions = []

    def evaluate(self, output_result_file="") -> List[float]:  # type: ignore
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            coco_predictions, coco_ids = self._coco_predictions()
            if output_result_file:
                json.dump(coco_predictions, open(output_result_file, "w"), indent=4)
                results = self.coco_api.loadRes(output_result_file)
            else:
                with NamedTemporaryFile(
                    prefix="coco_", suffix=".json", delete=False, mode="w"
                ) as tmpfile:
                    json.dump(coco_predictions, tmpfile, indent=4)
                results = self.coco_api.loadRes(tmpfile.name)
                try:
                    os.unlink(tmpfile.name)
                except OSError:
                    pass
            coco_eval = COCOeval(self.coco_api, results, "bbox")
            coco_eval.params.imgIds = coco_ids  # score only ids we've used
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats
            if self.distributed:
                dist.broadcast(torch.tensor(stats, device=self.distributed_device), 0)
        else:
            raise NotImplementedError(
                "Distributed evaluation not implemented for CocoStatsEvaluator"
            )
            stats = torch.tensor(0, device=self.distributed_device)
            dist.broadcast(stats, 0)
            stats = stats.item()
        self.reset()
        return cast(List, stats)
