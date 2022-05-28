from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from theseus.plotting.display import save_fig


def plot_clustering_results(
    features: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    show: bool = False,
) -> None:
    pca = PCA(2)
    compressed = pca.fit_transform(features)
    compressed = pd.DataFrame(compressed)
    compressed['labels'] = labels
    compressed.columns = [
        'x',
        'y',
    ]

    fig, ax = plt.subplots()
    ax.set(
        xlabel='x',
        ylabel='y',
        title='Clustering results',
    )

    for label in compressed['labels'].unique():
        cur_cluster = compressed[compressed['labels'] == label]
        ax.scatter(
            cur_cluster['x'].values,
            cur_cluster['y'].values,
            label=label,
        )

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(
            1.04,
            1,
        ),
    )

    save_fig(
        out_path,
        show,
        fig,
    )
