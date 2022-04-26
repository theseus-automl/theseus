from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

_DPI = 300
_ROTATION = 90


def plot_class_distribution(
    labels: pd.Series,
    out_path: Path,
    show: bool = False,
) -> None:
    count = len(labels)

    count_ax = sns.countplot(x=labels)
    plt.setp(
        count_ax.get_xticklabels(),
        rotation=_ROTATION,
    )
    count_ax.set(
        xlabel='Classes',
        ylabel='Count',
        title='Classes distribution',
    )

    freq_ax = count_ax.twinx()
    freq_ax.set_xlabel('Frequency [%]')

    count_ax.yaxis.tick_left()
    count_ax.yaxis.set_label_position('left')
    freq_ax.yaxis.tick_right()
    freq_ax.yaxis.set_label_position('right')

    for patch in count_ax.patches:
        x = patch.get_bbox().get_points()[:, 0].mean()  # noqa: WPS111
        y = patch.get_bbox().get_points()[1, 1]  # noqa: WPS111
        coords = (x, y)

        count_ax.annotate(
            f'{100 * y / count:.1f}%',
            coords,
            ha='center',
            va='bottom',
        )

    freq_ax.set_ylim(0, 100)
    count_ax.set_ylim(0, count)

    freq_ax.grid(None)

    plt.savefig(
        out_path,
        dpi=_DPI,
        bbox_inches='tight',
    )

    if show:
        plt.show()

    plt.close()
