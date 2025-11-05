from typing_extensions import Literal
import uuid
from copy import deepcopy
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_img_pred_subplots(
    img_list,
    bboxes_list,
    scores_list=None,
    labels_list=None,
    descriptions=None,
    coords: Literal["xyxy", "yxyx", "xywh"] = "xyxy",
    save_dir: Union[str, bool] = False,
    figsize: tuple = (15, 15),
):
    """
    Plot the predicted bounding boxes on multiple images in subplots.

    Args:
        imgs: A list of image tensors to plot.
        bboxes_list: A list of bounding boxes for each image.
        scores_list: A list of scores for each bounding box in each image.
        labels_list: A list of labels for each bounding box in each image.
        coords: The format of the bounding boxes.
        save_dir: If a string is provided, the plot will be saved to this directory.
                  If False, the plot will not be saved.
    """

    imgs = img_list
    num_imgs = len(imgs)
    cols = int(np.ceil(np.sqrt(num_imgs)))
    rows = int(np.ceil(num_imgs / cols))

    class_colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "lime",
        "pink",
    ]

    plt.figure(figsize=figsize)
    for idx, img_tensor in enumerate(imgs):
        plt.subplot(rows, cols, idx + 1)
        bboxes = bboxes_list[idx]
        scores = scores_list[idx] if scores_list is not None else None
        labels = labels_list[idx] if labels_list is not None else None

        if type(img_tensor) is str:
            img_tensor = plt.imread(img_tensor)
            img_tensor = np.transpose(img_tensor, (2, 0, 1))
        try:
            img_tensor = deepcopy(img_tensor).cpu().numpy()  # type: ignore
        except Exception:
            pass

        try:
            bboxes = deepcopy(bboxes).cpu().numpy()
        except Exception:
            pass

        try:
            if scores is not None:
                scores = deepcopy(scores).cpu().numpy()
        except Exception:
            pass

        try:
            if labels is not None:
                labels = deepcopy(labels).cpu().numpy()
        except Exception:
            pass

        plt.imshow(np.transpose(img_tensor, (1, 2, 0)))
        if descriptions is not None:
            plt.title(descriptions[idx])
        for i, bbox in enumerate(bboxes):
            if coords == "yxyx":
                ymin, xmin, ymax, xmax = bbox
            elif coords == "xyxy":
                xmin, ymin, xmax, ymax = bbox
            elif coords == "xywh":
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
            else:
                raise ValueError("coords must be one of 'xyxy', 'yxyx', or 'xywh'")

            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin

            plt.gca().add_patch(Rectangle((x, y), w, h, fill=False, color=class_colors[labels[i] - 1]))  # type: ignore
            if scores is not None:
                plt.text(
                    x,
                    y,
                    f"{scores[i]:.2f}",
                    bbox={"facecolor": class_colors[labels[i] - 1], "alpha": 0.1},  # type: ignore
                    color="white",
                )
            if labels is not None:
                plt.text(
                    x + w,
                    y,
                    f"{labels[i]}",
                    bbox={"facecolor": class_colors[labels[i] - 1], "alpha": 0.5},  # type: ignore
                    color="white",
                )

        plt.axis("off")
    plt.tight_layout()
    if type(save_dir) is not bool:
        plt.savefig(f"{save_dir}/pred_subplot_{uuid.uuid4()}.png")

    plt.show()


def plot_img_pred(
    img_tensor,
    bboxes,
    coords: Literal["xyxy", "yxyx", "xywh"],
    scores=None,
    labels=None,
    save_dir: Union[str, bool] = False,
    figsize: tuple = (10, 10),
    color="red",
):
    """
    Plot the predicted bounding boxes on the image.

    Args:
        img_tensor: The image tensor to plot.
        bboxes: The bounding boxes to plot on the image.
        yxyx: If True, the bounding boxes are in (ymin, xmin, ymax, xmax) format.
               If False, the bounding boxes are in (xmin, ymin, xmax, ymax) format.
        save_dir: If a string is provided, the plot will be saved to this directory.
                  If False, the plot will not be saved.
    """

    plt.figure(figsize=figsize)
    if type(img_tensor) is str:
        img_tensor = plt.imread(img_tensor)
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
    try:
        img_tensor = deepcopy(img_tensor).cpu().numpy()  # type: ignore
    except Exception:
        pass

    try:
        bboxes = deepcopy(bboxes).cpu().numpy()
    except Exception:
        pass

    try:
        if scores is not None:
            scores = deepcopy(scores).cpu().numpy()
    except Exception:
        pass

    try:
        if labels is not None:
            labels = deepcopy(labels).cpu().numpy()
    except Exception:
        pass

    plt.imshow(np.transpose(img_tensor, (1, 2, 0)))
    for i, bbox in enumerate(bboxes):
        if coords == "yxyx":
            ymin, xmin, ymax, xmax = bbox
        elif coords == "xyxy":
            xmin, ymin, xmax, ymax = bbox
        elif coords == "xywh":
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
        else:
            raise ValueError("coords must be one of 'xyxy', 'yxyx', or 'xywh'")

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        plt.gca().add_patch(Rectangle((x, y), w, h, fill=False, color=color))
        if scores is not None:
            plt.text(
                x,
                y,
                f"{scores[i]:.2f}",
                bbox={"facecolor": color, "alpha": 0.5},
                color="white",
            )
        if labels is not None:
            plt.text(
                x + w,
                y,
                f"{labels[i]}",
                bbox={"facecolor": color, "alpha": 0.5},
                color="white",
            )
    plt.axis("off")
    plt.show()

    if type(save_dir) is not bool:
        plt.savefig(f"{save_dir}/pred_{uuid.uuid4()}.png")
