import os
import hashlib
from typing import Union, List

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cache",
)


def dataframe_hash(df: Union[List, np.ndarray, pd.DataFrame]) -> str:
    if isinstance(df, np.ndarray):
        data = df.tobytes()
    elif isinstance(df, pd.DataFrame):
        data = df.values.tobytes()
    else:
        data = pd.DataFrame(df).values.tobytes()
    return hashlib.md5(data).hexdigest()


def get_embedding_positions(
        embeddings: Union[List[List[float]], np.ndarray, pd.DataFrame],
        pca_size: int = 100,
        read_cache: bool = True,
        write_cache: bool = True,
):
    if read_cache or write_cache:
        filename = dataframe_hash(embeddings)
        filename = f"{filename}-{pca_size}.csv"
        filename = os.path.join(
            CACHE_DIR, "df", filename,
        )
        if read_cache and os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
            return df

    # ---

    if len(embeddings) >= pca_size and len(embeddings[0]) >= pca_size:
        solver = PCA(pca_size)
        embeddings = solver.fit_transform(embeddings)

    solver = TSNE(init="pca")
    positions = solver.fit_transform(embeddings)
    df = pd.DataFrame(positions, columns=["x", "y"])

    # ---

    if write_cache:
        cache_dir = os.path.join(CACHE_DIR, "df")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        df.to_csv(filename)

    return df
