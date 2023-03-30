import pickle
from pathlib import Path
from sys import argv

from theseus.accelerator import Accelerator
from theseus.classification.auto import AutoClassifier
from theseus.dataset.text_dataset import TextDataset
from theseus.lang_code import LanguageCode


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
    dataset = load_toxic_comments(Path('/Users/tkasimov/Downloads/dataset.txt'))
    # with open('../dataset.pkl', 'rb') as f:
    #     dataset = pickle.load(f)
    
    import numpy as np
    print(np.unique(dataset.le.inverse_transform(dataset.labels), return_counts=True))

    print(len(dataset))

    # acc = Accelerator()
    # clf = AutoClassifier(
    #     Path('research/russian-toxic-comments-balanced'),
    #     acc,
    #     target_lang=LanguageCode.RUSSIAN,
    #     use_fasttext=True,
    #     use_tf_idf=False,
    #     use_bert=False,
    #     tf_idf_n_iter=50,
    #     tf_idf_n_jobs=-1,
    #     fast_text_n_iter=10,
    #     fast_text_n_jobs=4,
    #     ignore_imbalance=True,
    # )
    # clf.fit(dataset)
