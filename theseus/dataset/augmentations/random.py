from random import randint

from theseus.dataset.augmentations._abc import FillMaskAugmenter


class RandomReplacementAugmenter(FillMaskAugmenter):
    def __call__(
        self,
        text: str,
    ) -> str:
        tokens = text.split()

        idx = randint(
            1,
            len(tokens) - 1,
        )
        orig_word = tokens[idx]
        tokens[idx] = self._mask_token
        tokens = ' '.join(tokens)
        augmentations = self._pipeline(tokens)

        for res in augmentations:
            if res['token_str'] != orig_word:
                return res['sequence'].replace(self._cls_token, '').replace(self._sep_token, '').strip()

        return text


class RandomInsertionAugmenter(FillMaskAugmenter):
    def __call__(
        self,
        text: str,
    ) -> str:
        tokens = text.split()

        idx = randint(1, len(tokens) - 2)
        masked = ' '.join(tokens[:idx] + [self._mask_token] + tokens[idx:])
        augmented_text = self._pipeline(masked)[0]['sequence']
        augmented_text = augmented_text.replace(self._cls_token, '').replace(self._sep_token, '').strip()

        return augmented_text
