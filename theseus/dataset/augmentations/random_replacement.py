from random import randint
from types import MappingProxyType
from typing import NoReturn

from transformers import pipeline

from theseus.lang_code import LanguageCode

RANDOM_REPLACEMENT_MODELS = MappingProxyType({
    # TODO: use uncased models, you moron
    LanguageCode.ENGLISH: 'bert-base-cased',
    LanguageCode.SPANISH: 'PlanTL-GOB-ES/roberta-base-bne',
    LanguageCode.FRENCH: 'camembert-base',
    LanguageCode.GERMAN: 'bert-base-german-cased',
    LanguageCode.CHINESE: 'bert-base-chinese',
    LanguageCode.WU_CHINESE: 'bert-base-chinese',
    LanguageCode.YUE_CHINESE: 'bert-base-chinese',
    LanguageCode.SWEDISH: 'KB/bert-base-swedish-cased',
    LanguageCode.FINNISH: 'TurkuNLP/bert-base-finnish-cased-v1',
    LanguageCode.JAPANESE: 'cl-tohoku/bert-base-japanese-whole-word-masking',
    LanguageCode.ARABIC: 'aubmindlab/bert-base-arabertv02',
    LanguageCode.PORTUGUESE: 'neuralmind/bert-base-portuguese-cased',
    LanguageCode.ITALIAN: 'indigo-ai/BERTino',
    LanguageCode.TURKISH: 'dbmdz/electra-base-turkish-mc4-cased-generator',
    LanguageCode.DUTCH: 'GroNLP/bert-base-dutch-cased',
    LanguageCode.UKRAINIAN: 'youscan/ukr-roberta-base',
    LanguageCode.INDONESIAN: 'indolem/indobert-base-uncased',
    LanguageCode.HINDI: 'neuralspace-reverie/indic-transformers-hi-bert',
    LanguageCode.KOREAN: 'klue/bert-base',
    LanguageCode.PERSIAN: 'HooshvareLab/albert-fa-zwnj-base-v2',

    # correct models
    LanguageCode.DANISH: 'Maltehb/danish-bert-botxo',
    LanguageCode.POLISH: 'dkleczek/bert-base-polish-uncased-v1',
    LanguageCode.ROMANIAN: 'Geotrend/bert-base-ro-cased',  # uncased model is not available
    LanguageCode.NORWEGIAN: 'NbAiLab/nb-bert-base',  # uncased model is not available
    LanguageCode.CATALAN: 'PlanTL-GOB-ES/roberta-base-ca',
    LanguageCode.GREEK: 'nlpaueb/bert-base-greek-uncased-v1',
    LanguageCode.VIETNAMESE: 'nguyenvulebinh/envibert',
    LanguageCode.BULGARIAN: 'iarfmoose/roberta-base-bulgarian',
    LanguageCode.ICELANDIC: 'mideind/IceBERT-igc',
    LanguageCode.CZECH: 'ufal/robeczech-base',
    LanguageCode.HEBREW: 'onlplab/alephbert-base',
    LanguageCode.BENGALI: 'sagorsarker/bangla-bert-base',
    LanguageCode.ESPERANTO: 'julien-c/EsperBERTo-small',
    LanguageCode.MARATHI: 'DarshanDeshpande/marathi-distilbert',
    LanguageCode.CROATIAN: 'EMBEDDIA/crosloengual-bert',
    LanguageCode.SLOVENIAN: 'EMBEDDIA/sloberta',
    LanguageCode.LITHUANIAN: 'EMBEDDIA/litlat-bert',
    LanguageCode.ESTONIAN: 'tartuNLP/EstBERT',
    LanguageCode.THAI: 'airesearch/wangchanberta-base-att-spm-uncased',
    LanguageCode.URDU: 'urduhack/roberta-urdu-small',
    LanguageCode.SVAHILI: 'castorini/afriberta_large',
    LanguageCode.YORUBA: 'castorini/afriberta_large',
    LanguageCode.ASSAMESE: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.TAMIL: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.ORIAY: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.AFRIKAANS: 'jannesg/takalane_afr_roberta',
    LanguageCode.LATVIAN: 'EMBEDDIA/litlat-bert',
    LanguageCode.SLOVAK: 'EMBEDDIA/litlat-bert',
    LanguageCode.BELARUSIAN: 'KoichiYasuoka/roberta-small-belarusian',
    LanguageCode.BASQUE: 'ixa-ehu/roberta-eus-euscrawl-base-cased',
    LanguageCode.GUJARATI: 'surajp/RoBERTa-hindi-guj-san',
    LanguageCode.TAGALOG: 'jcblaise/roberta-tagalog-base',

})


class RandomReplacementAugmenter:
    def __init__(
        self,
    ) -> NoReturn:
        self._unmasker = pipeline(
            'fill-mask',
            model='bert-base-multilingual-cased',
        )
        self._cls_token = self._unmasker.tokenizer.cls_token
        self._mask_token = self._unmasker.tokenizer.mask_token
        self._sep_token = self._unmasker.tokenizer.sep_token

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
        augmentations = self._unmasker(tokens)

        for res in augmentations:
            if res['token_str'] != orig_word:
                augmented_text = res['sequence'].replace(self._cls_token, '').replace(self._sep_token, '').strip()
                break
        else:
            raise ValueError('unable to generate output different from input')

        return augmented_text
