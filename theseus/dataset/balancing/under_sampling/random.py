import numpy as np
import pandas as pd

from theseus.dataset.balancing.sampler_mixin import SamplerMixin


class RandomUnderSampler(SamplerMixin):
    def __init__(
        self,
    ) -> None:
        # TODO: random state for reproducibility
        pass

    def __call__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        df, counts, target_samples = super().__call__(
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
