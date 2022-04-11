from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# from matplotlib import ticker


def plot_class_distribution(
    data: pd.Series,
    out_path: Path,
    show: bool = False,
) -> None:
    count = len(data)

    count_ax = sns.countplot(x=data)
    plt.setp(
        count_ax.get_xticklabels(),
        rotation=90,
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

    for p in count_ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]

        count_ax.annotate(
            f'{100 * y / count:.1f}%',
            (
                x.mean(),
                y,
            ),
            ha='center',
            va='bottom',
        )

    # count_ax.yaxis.set_major_locator(ticker.LinearLocator(10))
    freq_ax.set_ylim(0, 100)
    count_ax.set_ylim(0, count)
    # freq_ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    freq_ax.grid(None)

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches='tight',
    )

    if show:
        plt.show()

    plt.close()
