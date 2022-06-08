from itertools import repeat
from pathlib import Path
from sys import argv
from timeit import default_timer as timer

import pandas as pd

from theseus.accelerator import Accelerator
from theseus.dataset.text_dataset import TextDataset
from theseus.zero_shot.auto import AutoZeroShotClassifier


def load_imdb_subdir(
    _path: Path,
) -> list:
    _reviews = []

    for _file in _path.iterdir():
        with open(_file, 'r', encoding='utf-8') as _f:
            _reviews.append(_f.read().replace('\n', ''))

    return _reviews


def load_imdb(
    _base_dir: Path,
) -> pd.DataFrame:
    _pos = load_imdb_subdir(_base_dir / 'train' / 'pos') + load_imdb_subdir(_base_dir / 'test' / 'pos')
    _neg = load_imdb_subdir(_base_dir / 'train' / 'neg') + load_imdb_subdir(_base_dir / 'test' / 'neg')

    _df = pd.DataFrame()
    _df['text'] = _pos + _neg
    _df['label'] = list(repeat(1, len(_pos))) + list(repeat(0, len(_neg)))

    return _df


if __name__ == '__main__':
    df = load_imdb(Path(argv[1]))
    base_dir = Path(argv[2])
    batch_size = Path(argv[3])
    accelerator = Accelerator(gpus=0)

    candidate_labels = [
        [
            'positive',
            'negative',
        ],
        [
            'positive review',
            'negative review',
        ],
        [
            'positive movie review',
            'negative movie review',
        ],
    ]

    for label_set in candidate_labels:
        clf = AutoZeroShotClassifier(
            base_dir / ' '.join(label_set),
            accelerator,
            label_set,
        )
        dataset = TextDataset(df['text'].tolist())

        start = timer()
        clf.fit(dataset)
        end = timer()

        print(f'elapsed time {end - start} seconds')

        del clf
