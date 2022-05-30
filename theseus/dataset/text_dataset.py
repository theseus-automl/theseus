from typing import (
    Optional,
    Sequence,
    Tuple,
    Union,
)

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        if labels is not None:
            if len(texts) != len(labels):
                raise ValueError(f'length mismatch: found {len(texts)} texts and {len(labels)} labels')

            self.le = LabelEncoder()
            self.labels = self.le.fit_transform(labels)
        else:
            self.labels = None

        self.texts = list(texts)
        self.class_weights = None

    def __len__(
        self,
    ) -> int:
        return len(self.texts)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[str, Union[int, list]]:
        label = [] if self.labels is None else self.labels[idx]

        return (
            self.texts[idx],
            label,
        )

    @property
    def num_labels(
        self,
    ) -> Optional[int]:
        return None if self.labels is None else len(set(self.labels))
