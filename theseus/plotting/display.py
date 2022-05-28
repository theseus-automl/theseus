from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from theseus.plotting.defaults import Defaults


def save_fig(
    out_path: Path,
    show: bool,
    fig: Optional[Figure] = None,
) -> None:
    used_fig = plt if fig is None else fig

    used_fig.savefig(
        out_path,
        dpi=Defaults.dpi,
        bbox_inches='tight',
    )

    if show:
        used_fig.show()

    if isinstance(used_fig, Figure):
        plt.close(used_fig)
    else:
        plt.close()
