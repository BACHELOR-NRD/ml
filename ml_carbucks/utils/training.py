from pathlib import Path
from dataclasses import dataclass

from matplotlib import pyplot as plt
import pandas as pd


@dataclass
class ResultSaver:
    res_dir: str | Path
    name: str
    data: list = []
    metadata: dict = {}

    def __post_init__(self):
        Path(self.res_dir).mkdir(parents=True, exist_ok=True)

    def save(self, epoch: int, loss: float, val_map: float, **kwargs) -> None:
        self.data.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_map": val_map,
                **kwargs,
                **self.metadata,
            }
        )
        pd.DataFrame(self.data).to_csv(
            Path(self.res_dir) / f"{self.name}.csv", index=False
        )

    def plot(self, secondary_y: str = "val_map", save: bool = True) -> None:
        df = pd.DataFrame(self.data)
        df.plot(
            x="epoch",
            y=["loss", secondary_y],
            secondary_y=secondary_y,  # type: ignore
            title="Training Metrics over Epochs",
        )
        if save:
            plt.savefig(Path(self.res_dir) / f"{self.name}_plot.png")
        plt.show()
