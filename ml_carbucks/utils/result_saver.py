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
    append: bool = True

    def __post_init__(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)
        if self.append:
            existing_file = Path(self.path) / f"{self.name}.csv"
            if existing_file.exists():
                existing_data = pd.read_csv(existing_file)
                self.data = existing_data.to_dict(orient="records")

    def save(self, **kwargs) -> "ResultSaver":

        self.data.append(
            {
                **self.metadata,
                **kwargs,
            }
        )
        pd.DataFrame(self.data).to_csv(
            Path(self.path) / f"{self.name}.csv", index=False
        )

        return self

    def plot(
        self,
        x: str = "epoch",
        y: str = "loss",
        secondaries_y: list[str] = ["val_map_50_95"],
        save: bool = True,
        show: bool = True,
        title: str = "Training Metrics over Epochs",
    ) -> "ResultSaver":
        df = pd.DataFrame(self.data)
        df.plot(
            x=x,
            y=[y, *secondaries_y],
            secondary_y=secondaries_y,  # type: ignore
            title=title,
        )
        if save:
            plt.savefig(Path(self.path) / f"{self.name}_plot.png")

        if show:
            plt.show()
        plt.close()

        return self
