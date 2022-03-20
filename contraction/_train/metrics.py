import json
from pathlib import Path
from typing import List, Optional, Union


class Metrics:
    def __init__(self):
        self._metric_map = {}

    def log(self, metric_name: str, metric_value: Union[float, int, str]):
        if metric_name not in self._metric_map:
            self._metric_map[metric_name] = []
        self._metric_map[metric_name].append(metric_value)

    def get(self, metric_name: str) -> List[Union[float, int, str]]:
        return self._metric_map[metric_name]

    def save(self, filepath: Path, indent: Optional[int] = 2):
        with filepath.open('w') as f:
            json.dump(self._metric_map, f, indent=indent)

    @classmethod
    def load(cls, filepath: Path):
        metrics = cls()
        with filepath.open() as f:
            metrics._metric_map = json.load(f)
        return metrics
