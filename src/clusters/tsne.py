import os
import hashlib

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_embedding_positions(
        embeddings,
        pca_size: int = 100,
        read_cache: bool = True,
        write_cache: bool = True,
):
    if read_cache or write_cache:
        filename = hashlib.md5(pd.DataFrame(embeddings).to_string().encode("utf-8")).hexdigest()
        filename = f"{filename}-{pca_size}.csv"
        filename = os.path.join(
            "cache", "df", filename,
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
        if not os.path.exists("cache/df"):
            os.makedirs("cache/df")
        df.to_csv(filename)

    return df
