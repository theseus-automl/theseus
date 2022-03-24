import numpy as np
import pandas as pd

from theseus.dataset.balancing._common import prepare


class RandomUnderSampler:
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
                to_drop = np.random.choice(
                    df[df['labels'] == label].index,
                    abs(n_samples - target_samples),
                    replace=False,
                )
                df = df.drop(to_drop)

        return df
