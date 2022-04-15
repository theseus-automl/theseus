import random
from typing import (
    NoReturn,
    Optional,
)

import numpy as np
import torch


def seed_everything(
    seed: Optional[int],
) -> None:
    if seed is None:
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min

        seed = random.randint(
            min_seed_value,
            max_seed_value,
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
