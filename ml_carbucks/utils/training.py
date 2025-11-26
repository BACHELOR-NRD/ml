from typing import Literal, Optional
import torch
from timm.scheduler.scheduler import Scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

from ml_carbucks.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_batches: int,
    epochs: int,
    accumulation_steps: int,
    lr: float,
    scheduler: Optional[Literal["cosine"]],
) -> Optional[Scheduler]:

    if scheduler is None:
        final_scheduler = None
    elif scheduler == "cosine":
        if epochs < 50:
            error_msg = "Using cosine scheduler with less than 50 epochs may lead to suboptimal results."
            logger.error(error_msg)
            raise ValueError(error_msg)

        total_steps = epochs * num_batches // accumulation_steps
        final_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=total_steps,
            lr_min=lr * 0.05,
            t_in_epochs=False,
            warmup_lr_init=lr * 0.1,  # type: ignore
            warmup_t=int(total_steps * 0.1),
            warmup_prefix=False,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")

    return final_scheduler
