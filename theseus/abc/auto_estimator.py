from pathlib import Path
from typing import Optional

from pytorch_lightning import seed_everything

from theseus import Accelerator
from theseus.defaults import RANDOM_STATE
from theseus.lang_code import LanguageCode


class AutoEstimator:
    def __init__(
        self,
        out_dir: Path,
        accelerator: Accelerator,
        target_lang: Optional[LanguageCode] = None,
    ) -> None:
        seed_everything(RANDOM_STATE)

        self._out_dir = out_dir
        self._accelerator = accelerator
        self._target_lang = target_lang

        self._out_dir.mkdir(
            exist_ok=True,
            parents=True,
        )
