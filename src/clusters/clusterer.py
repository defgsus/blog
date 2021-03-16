from copy import copy, deepcopy
from typing import Tuple, Union, Callable

import numpy as np


class LiveClusterer:
    """
    Some nearest-means like clustering that
    clusters data without storing all single data entries.
    """
    def __init__(
            self,
            distance_metric: Union[str, Callable] = "abs",
            normalize_by_length: bool = True,
    ):
        self.clusters = []
        self.normalize_by_length = True
        self.distance_metric = distance_metric
        if callable(self.distance_metric):
            self._distance_func = self.distance_metric
        elif self.distance_metric == "abs":
            self._distance_func = lambda a, b: np.sum(np.abs(a - b))
        elif self.distance_metric.startswith("euclid"):
            self._distance_func = lambda a, b: np.sqrt(np.sum(a - b))

    def __copy__(self):
        clusterer = self.__class__(
            distance_metric=self.distance_metric,
            normalize_by_length=self.normalize_by_length,
        )
        clusterer.clusters = deepcopy(self.clusters)
        return clusterer

    def add(
            self,
            key, data: np.array,
            threshold: float = 1.,
            force_match: bool = False,
    ):
        best_match, best_diff = self.get_best_match(data)

        if not best_match or (best_diff > threshold and not force_match):
            self.clusters.append({
                "count": 1,
                "data": data,
                "keys": {key},
            })
        else:
            self._add_to_cluster(best_match, data, {key})

    def get_best_match(self, data: np.array) -> Tuple[dict, float]:
        best_match = None
        best_diff = None
        for cl in self.clusters:
            diff = self._distance_func(cl["data"], data)
            if self.normalize_by_length:
                diff /= len(data)
            if best_diff is None or diff < best_diff:
                best_diff, best_match = diff, cl
        return best_match, best_diff

    def recluster_below_count(self, count: int):
        clusterer = copy(self)
        clusterer.clusters = []
        small_clusters = []
        for cl in self.clusters:
            if cl["count"] >= count:
                clusterer.clusters.append(cl)
            else:
                small_clusters.append(cl)

        print(f"reclustering {len(small_clusters)} clusters below count of {count}")
        for cl in small_clusters:
            best_match, best_diff = self.get_best_match(cl["data"])
            self._add_to_cluster(best_match, cl["data"], cl["keys"])

        return clusterer

    def _add_to_cluster(self, cluster: dict, data: dict, keys: set):
        cluster["data"] = (cluster["data"] * cluster["count"] + data) / (cluster["count"] + 1)
        cluster["count"] += 1
        cluster["keys"] |= keys
