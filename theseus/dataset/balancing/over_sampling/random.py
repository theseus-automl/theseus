import numpy as np
import pandas as pd

from theseus.dataset.balancing._common import prepare


class RandomOverSampler:
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        df, counts, target_samples = prepare(
            texts,
            labels,
            'under',
        )

        for label, n_samples in counts.items():
            if n_samples != target_samples:
                sampled = df[df['labels'] == label].sample(
                    n=abs(n_samples - target_samples),
                    replace=False,
                )
                df = pd.concat(
                    [
                        df,
                        sampled,
                    ],
                    ignore_index=True,
                )

        return df
