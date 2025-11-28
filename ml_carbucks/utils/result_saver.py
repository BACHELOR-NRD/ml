from pathlib import Path
from dataclasses import dataclass, field

from matplotlib import pyplot as plt
import pandas as pd


@dataclass
class ResultSaver:
    path: str | Path
    name: str
    data: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def save(
        self, epoch: int, loss: float, val_map_50_95: float, **kwargs
    ) -> "ResultSaver":
        self.data.append(
            {
                "epoch": epoch,
                "loss": loss,
                "val_map_50_95": val_map_50_95,
                **kwargs,
                **self.metadata,
            }
        )
        pd.DataFrame(self.data).to_csv(
            Path(self.path) / f"{self.name}.csv", index=False
        )

        return self

    def plot(
        self,
        secondaries_y: list[str] = ["val_map_50_95"],
        save: bool = True,
        show: bool = True,
    ) -> "ResultSaver":
        df = pd.DataFrame(self.data)
        df.plot(
            x="epoch",
            y=["loss", *secondaries_y],
            secondary_y=secondaries_y,  # type: ignore
            title="Training Metrics over Epochs",
        )
        if save:
            plt.savefig(Path(self.path) / f"{self.name}_plot.png")

        if show:
            plt.show()
        plt.close()

        return self
