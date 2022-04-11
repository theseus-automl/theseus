from argparse import (
    ArgumentParser,
    SUPPRESS,
)

from tqdm import tqdm
from transformers.utils.logging import set_verbosity_error

from theseus.dataset.augmentations.generation import GPTAugmenter
from theseus.dataset.augmentations.back_translation import BackTranslationAugmenter
from theseus.dataset.augmentations.random_insertion import RandomInsertionAugmenter
from theseus.dataset.augmentations.random_replacement import RandomReplacementAugmenter


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
        BackTranslationAugmenter,
        GPTAugmenter,
        RandomInsertionAugmenter,
        RandomReplacementAugmenter,
    ]

    for model_class in tqdm(models, disable=not verbose, desc='downloading models'):
        model = model_class()
        del model
