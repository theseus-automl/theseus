import gc
from argparse import (
    ArgumentParser,
    SUPPRESS,
)

import torch
from tqdm import tqdm
from transformers.utils.logging import set_verbosity_error

from theseus.dataset.augmentations._models import (
    BACK_TRANSLATION_MODELS,
    FILL_MASK_MODELS,
    GENERATION_MODELS,
)
from theseus.dataset.augmentations.back_translation import BackTranslationAugmenter
from theseus.dataset.augmentations.generation import GPTAugmenter
from theseus.dataset.augmentations.random import RandomInsertionAugmenter

if __name__ == '__main__':
    set_verbosity_error()

    parser = ArgumentParser(description='Download all models used by Theseus')
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=SUPPRESS,
        help='Display progress bar',
    )

    verbose = 'verbose' in parser.parse_args()
    models = [
        (
            BackTranslationAugmenter,
            BACK_TRANSLATION_MODELS,
        ),
        (
            GPTAugmenter,
            GENERATION_MODELS,
        ),
        (
            RandomInsertionAugmenter,
            FILL_MASK_MODELS,
        ),
    ]

    for model_class, supported_models in tqdm(models, disable=not verbose, desc='downloading models'):
        for lang in supported_models:
            model = model_class(lang)

            del model
            gc.collect()
            torch.cuda.empty_cache()
