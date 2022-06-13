from pathlib import Path
from sys import argv

from sklearn.datasets import fetch_20newsgroups

from theseus import Accelerator
from theseus.clustering.auto import AutoClusterer
from theseus.dataset.text_dataset import TextDataset

if __name__ == '__main__':
    texts = fetch_20newsgroups(subset='all')['data']
    texts = [text.replace('\n', ' ') for text in texts]

    acc = Accelerator()
    clusterer = AutoClusterer(
        Path(argv[1]),
        acc,
        use_tf_idf=True,
        use_fasttext=True,
        use_bert=True,
        fast_text_n_jobs=-1,
        fast_text_n_iter=10,
        tf_idf_n_jobs=-1,
        tf_idf_n_iter=10,
        sbert_n_jobs=2,
        sbert_n_iter=10,
    )
    clusterer.fit(TextDataset(texts))
