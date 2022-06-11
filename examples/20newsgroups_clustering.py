from pathlib import Path
from sys import argv

from sklearn.datasets import fetch_20newsgroups

from theseus import Accelerator
from theseus.clustering.auto import AutoClusterer


if __name__ == '__main__':
    texts = fetch_20newsgroups(subset='all')
    acc = Accelerator(gpus=0)
    clusterer = AutoClusterer(
        Path(argv[1]),
        acc,
        use_tf_idf=True,
        use_fasttext=False,
        use_bert=False,
        fast_text_n_jobs=-1,
        fast_text_n_iter=1,
        tf_idf_n_jobs=-1,
        tf_idf_n_iter=1,
    )
