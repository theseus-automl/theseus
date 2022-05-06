from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# from matplotlib import ticker
from matplotlib.figure import Figure
from sklearn.metrics._scorer import _PredictScorer

from theseus.log import setup_logger

_logger = setup_logger(__name__)


class _Defaults:
    dpi = 300
    rotation = 90

    colors = (
        'green',
        'black',
        'red',
        'yellow',
        'cyan',
        'magenta',
    )
    train_test_styles = (
        (
            'train',
            '--',
        ),
        (
            'test',
            '-',
        ),
    )

    annotation_offset = 0.005

    @staticmethod
    def get_fill_alpha(
        split: str,
    ) -> float:
        return 0.1 if split == 'test' else 0

    @staticmethod
    def get_plot_alpha(
        split: str,
    ) -> float:
        return 1 if split == 'test' else 0.7


def plot_class_distribution(
    labels: pd.Series,
    out_path: Path,
    show: bool = False,
) -> None:
    count = len(labels)

    count_ax = sns.countplot(x=labels)
    plt.setp(
        count_ax.get_xticklabels(),
        rotation=_Defaults.rotation,
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

    _save_fig(
        out_path,
        show,
    )


def plot_gs_result(
    gs_result: Dict[str, Any],
    scoring: Dict[str, _PredictScorer],
    out_path: Path,
    show: bool = False,
) -> None:
    for key in gs_result:
        if key.startswith('param_'):
            _plot_gs_result_by_single_param(
                gs_result,
                scoring,
                key,
                out_path / f'{key}.png',
                show,
            )


def plot_metrics(
    gs_result: Dict[str, Any],
    out_path: Path,
    show: bool = False,
) -> None:
    for split in ('train', 'test'):
        _plot_split_metrics(
            gs_result,
            split,
            out_path / f'{split}.png',
            show,
        )


def _plot_split_metrics(
    gs_result: Dict[str, Any],
    split: str,
    out_path: Path,
    show: bool = False,
) -> None:
    metrics = defaultdict(list)

    for key in gs_result:
        if key.startswith('split') and split in key:
            metrics[key.split('_')[-1]].append(key)

    fig, axes = plt.subplots(
        ncols=len(metrics),
        figsize=(40, 10),
    )
    fig.suptitle(f'Metrics for {split}')

    for metric, ax in zip(metrics, axes):
        ax.set(
            xlabel=metric,
            ylabel='Score',
        )
        # TODO: log scale does not work with formatter
        # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        ax.set_yscale('log')

        for key, color in zip(metrics[metric], _Defaults.colors):
            ax.plot(
                gs_result[key],
                color=color,
                label=f"split #{key.split('_')[0].replace('split', '')}",
            )

        ax.legend(loc='upper right')

    _save_fig(
        out_path,
        show,
    )


def _plot_gs_result_by_single_param(
    gs_result: Dict[str, Any],
    scoring: Dict[str, _PredictScorer],
    target_param: str,
    out_path: Path,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots()
    ax.set(
        xlabel=target_param.split('__')[1],
        ylabel='Score',
        title='GridSearchCV result',
    )
    ax.set_yscale('log')

    try:
        x_axis = np.array(
            gs_result[target_param].data,
            dtype=float,
        )
    except ValueError:
        x_axis = np.array(
            gs_result[target_param].data,
            dtype=object,
        )

    if len(x_axis) == 1:
        _logger.warning(f'param {target_param} has only 1 value, so the plot will be uninformative')

    for scorer, color in zip(sorted(scoring), _Defaults.colors):
        for sample, style in _Defaults.train_test_styles:
            sample_score_mean = gs_result[f'mean_{sample}_{scorer}']
            sample_score_std = gs_result[f'std_{sample}_{scorer}']
            ax.fill_between(
                x_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=_Defaults.get_fill_alpha(sample),
                color=color,
            )
            ax.plot(
                x_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=_Defaults.get_plot_alpha(sample),
                label=f'{scorer} ({sample})',
            )

        best_index = np.nonzero(gs_result[f'rank_test_{scorer}'] == 1)[0][0]
        best_score = gs_result[f'mean_test_{scorer}'][best_index]

        ax.plot(
            [
                x_axis[best_index],
                x_axis[best_index],
            ],
            [
                0,
                best_score,
            ],
            linestyle='-.',
            color=color,
            marker='x',
            markeredgewidth=3,
            ms=8,
        )

        ax.annotate(
            f'{best_score:.2f}',
            (
                x_axis[best_index],
                best_score + _Defaults.annotation_offset,
            ),
        )

    plt.legend(loc='best')
    plt.grid(False)

    _save_fig(
        out_path,
        show,
        fig,
    )


def _save_fig(
    out_path: Path,
    show: bool,
    fig: Optional[Figure] = None,
) -> None:
    used_fig = plt if fig is None else fig

    used_fig.savefig(
        out_path,
        dpi=_Defaults.dpi,
        bbox_inches='tight',
    )

    if show:
        used_fig.show()

    if isinstance(used_fig, Figure):
        plt.close(used_fig)
    else:
        plt.close()
