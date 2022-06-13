from pathlib import Path
from sys import argv

from theseus.accelerator import Accelerator
from theseus.classification.auto import AutoClassifier
from theseus.dataset.text_dataset import TextDataset


def load_toxic_comments(
    _path: Path,
) -> TextDataset:
    _texts = []
    _labels = []

    with open(_path, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            _cur_labels, _text = _line.split(
                ' ',
                1,
            )

            for _label in _cur_labels.split(','):
                _texts.append(_text)
                _labels.append(_label)

    return TextDataset(
        _texts,
        _labels,
    )


if __name__ == '__main__':
    dataset = load_toxic_comments(Path(argv[1]))
    acc = Accelerator(gpus=0)
    clf = AutoClassifier(
        Path(argv[2]),
        acc,
        use_fasttext=True,
        use_tf_idf=True,
        use_bert=True,
        tf_idf_n_iter=50,
        fast_text_n_iter=50,
        ignore_imbalance=True,
    )
    clf.fit(dataset)
