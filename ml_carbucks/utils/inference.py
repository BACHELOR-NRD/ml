from typing_extensions import Literal
import uuid
from copy import deepcopy
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_img_pred(
    img_tensor,
    bboxes,
    coords: Literal["xyxy", "yxyx", "xywh"],
    save_dir: Union[str, bool] = False,
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

    plt.figure(figsize=(10, 10))
    if type(img_tensor) is str:
        img_tensor = plt.imread(img_tensor)
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
    try:
        img_tensor = deepcopy(img_tensor).cpu().numpy()
    except Exception:
        pass

    try:
        bboxes = deepcopy(bboxes).cpu().numpy()
    except Exception:
        pass

    plt.imshow(np.transpose(img_tensor, (1, 2, 0)))
    for bbox in bboxes:
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

        plt.gca().add_patch(Rectangle((x, y), w, h, fill=False, color="red"))
    plt.axis("off")
    plt.show()

    if type(save_dir) is not bool:
        plt.savefig(f"{save_dir}/pred_{uuid.uuid4()}.png")
