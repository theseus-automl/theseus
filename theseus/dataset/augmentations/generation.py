import warnings
from random import randint
from types import MappingProxyType

from transformers import pipeline

from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.validators import Integer

GENERATION_MODELS = MappingProxyType({
    LanguageCode.ENGLISH: 'distilgpt2',
    LanguageCode.SPANISH: 'DeepESP/gpt2-spanish',
    LanguageCode.FRENCH: 'asi/gpt-fr-cased-base',
    LanguageCode.GERMAN: 'dbmdz/german-gpt2',
    LanguageCode.CHINESE: 'ckiplab/gpt2-base-chinese',
    LanguageCode.WU_CHINESE: 'ckiplab/gpt2-base-chinese',
    LanguageCode.YUE_CHINESE: 'ckiplab/gpt2-base-chinese',
    LanguageCode.SWEDISH: 'birgermoell/swedish-gpt',
    LanguageCode.FINNISH: 'Finnish-NLP/gpt2-large-finnish',
    LanguageCode.RUSSIAN: 'sberbank-ai/rugpt3large_based_on_gpt2',
    LanguageCode.JAPANESE: 'rinna/japanese-gpt-1b',
    LanguageCode.ARABIC: 'aubmindlab/aragpt2-large',
    LanguageCode.PORTUGUESE: 'pierreguillou/gpt2-small-portuguese',
    LanguageCode.ITALIAN: 'LorenzoDeMattei/GePpeTto',
    LanguageCode.TURKISH: 'redrussianarmy/gpt2-turkish-cased',
    LanguageCode.DUTCH: 'GroNLP/gpt2-small-dutch',
    LanguageCode.INDONESIAN: 'cahya/gpt2-small-indonesian-522M',
    LanguageCode.KOREAN: 'skt/kogpt2-base-v2',
    LanguageCode.PERSIAN: 'bolbolzaban/gpt2-persian',
    LanguageCode.POLISH: 'flax-community/papuGaPT2',
    LanguageCode.ROMANIAN: 'readerbench/RoGPT2-large',
    LanguageCode.NORWEGIAN: 'pere/norwegian-gpt2',
    LanguageCode.GREEK: 'nikokons/gpt2-greek',
    LanguageCode.VIETNAMESE: 'imthanhlv/gpt2news',
    LanguageCode.HUNGARIAN: 'NYTK/text-generation-news-gpt2-small-hungarian',
    LanguageCode.HEBREW: 'Norod78/hebrew-gpt_neo-small',
    LanguageCode.BENGALI: 'flax-community/gpt2-bengali',
    LanguageCode.MARATHI: 'l3cube-pune/marathi-gpt',
    LanguageCode.CZECH: 'MU-NLPC/CzeGPT-2',
    LanguageCode.CROATIAN: 'macedonizer/hr-gpt2',
    LanguageCode.SLOVENIAN: 'macedonizer/sl-gpt2',
    LanguageCode.THAI: 'flax-community/gpt2-base-thai',
    LanguageCode.SVAHILI: 'flax-community/gpt2-swahili',
    LanguageCode.TAMIL: 'abinayam/gpt-2-tamil',
    LanguageCode.SLOVAK: 'Milos/slovak-gpt-j-1.4B',
    LanguageCode.TAGALOG: 'jcblaise/gpt2-tagalog',
    LanguageCode.SINHALA: 'keshan/sinhala-gpt2-newswire',
    LanguageCode.SERBIAN: 'macedonizer/sr-gpt2',
    LanguageCode.MACEDONIAN: 'macedonizer/mk-gpt2',
    LanguageCode.MONGOLIAN: 'Ochiroo/tiny_mn_gpt',
    LanguageCode.JAVANESE: 'w11wo/javanese-gpt2-small-imdb',
    LanguageCode.SUNDANESE: 'w11wo/sundanese-gpt2-base',
    LanguageCode.ALBANIAN: 'macedonizer/al-gpt2',
})


class GPTAugmenterShortInputWarning(Warning):
    pass


class GPTAugmenter:
    _min_input_len = Integer(min_value=5)
    _max_sequences = Integer(min_value=1)

    def __init__(
        self,
        target_lang: LanguageCode,
        min_input_len: int = 5,
        max_sequences: int = 10,
    ) -> None:
        self._min_input_len = min_input_len
        self._max_sequences = max_sequences

        if target_lang not in GENERATION_MODELS:
            raise UnsupportedLanguageError(f'generation model not found for {target_lang} language')

        self._generator = pipeline(
            'text-generation',
            model=GENERATION_MODELS[target_lang],
        )

    def __call__(
        self,
        text: str,
    ) -> str:
        input_length = len(text.split())

        if input_length < self._min_input_len:
            warnings.warn(
                'Input is too short. Results may be inaccurate',
                GPTAugmenterShortInputWarning,
            )

        num_new_words = randint(input_length // 2, input_length)
        output = self._generator(
            text,
            max_length=input_length + num_new_words,
            num_return_sequences=self._max_sequences,
        )
        augmented_text = output[randint(0, self._max_sequences - 1)]['generated_text']

        return augmented_text
