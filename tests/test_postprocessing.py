import torch
import pytest

from ml_carbucks.utils.postprocessing import postprocess_prediction


@pytest.fixture
def setup_tensors():
    boxes = torch.tensor(
        [[10, 10, 20, 20], [11, 11, 21, 21], [100, 100, 120, 120]], dtype=torch.float32
    )
    scores = torch.tensor([0.9, 0.85, 0.8], dtype=torch.float32)
    labels = torch.tensor([1, 1, 2], dtype=torch.long)
    return boxes, scores, labels


@pytest.mark.postprocessing
def test_nsm_postprocessing_filters_by_score(setup_tensors):

    boxes, scores, labels = setup_tensors

    res = postprocess_prediction(
        boxes, scores, labels, conf_threshold=0.86, iou_threshold=0.5, max_detections=10
    )

    assert len(res["boxes"]) == 1
    assert torch.allclose(
        res["boxes"][0], torch.tensor([10, 10, 20, 20], dtype=torch.float32)
    )
    assert torch.isclose(res["scores"][0], torch.tensor(0.9, dtype=torch.float32))
    assert res["labels"][0].item() == 1


@pytest.mark.postprocessing
def test_nsm_postprocessing_iou_filtering(setup_tensors):

    boxes, scores, labels = setup_tensors

    res = postprocess_prediction(
        boxes, scores, labels, conf_threshold=0.0, iou_threshold=0.5, max_detections=10
    )

    assert len(res["boxes"]) == 2
    assert torch.allclose(
        res["boxes"][0], torch.tensor([10, 10, 20, 20], dtype=torch.float32)
    )
    assert torch.isclose(res["scores"][0], torch.tensor(0.9, dtype=torch.float32))
    assert res["labels"][0].item() == 1
    assert torch.allclose(
        res["boxes"][1], torch.tensor([100, 100, 120, 120], dtype=torch.float32)
    )
    assert torch.isclose(res["scores"][1], torch.tensor(0.8, dtype=torch.float32))
    assert res["labels"][1].item() == 2


@pytest.mark.postprocessing
def test_nsm_postprocessing_max_detections(setup_tensors):

    boxes, scores, labels = setup_tensors

    res = postprocess_prediction(
        boxes, scores, labels, conf_threshold=0.0, iou_threshold=0.5, max_detections=1
    )

    assert len(res["boxes"]) == 1
    assert torch.allclose(
        res["boxes"][0], torch.tensor([10, 10, 20, 20], dtype=torch.float32)
    )
    assert torch.isclose(res["scores"][0], torch.tensor(0.9, dtype=torch.float32))
    assert res["labels"][0].item() == 1
