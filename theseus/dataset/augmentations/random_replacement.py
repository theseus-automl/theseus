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
