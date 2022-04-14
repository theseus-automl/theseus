from random import randint
from types import MappingProxyType

from transformers import pipeline

from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode

RANDOM_REPLACEMENT_MODELS = MappingProxyType({
    LanguageCode.ENGLISH: 'bert-base-uncased',
    LanguageCode.SPANISH: 'PlanTL-GOB-ES/roberta-base-bne',
    LanguageCode.FRENCH: 'camembert-base',
    LanguageCode.GERMAN: 'dbmdz/bert-base-german-uncased',
    LanguageCode.CHINESE: 'bert-base-chinese',
    LanguageCode.WU_CHINESE: 'bert-base-chinese',
    LanguageCode.YUE_CHINESE: 'bert-base-chinese',
    LanguageCode.SWEDISH: 'KB/bert-base-swedish-cased',
    LanguageCode.FINNISH: 'TurkuNLP/bert-base-finnish-uncased-v1',
    LanguageCode.JAPANESE: 'cl-tohoku/bert-base-japanese-whole-word-masking',
    LanguageCode.ARABIC: 'aubmindlab/bert-base-arabertv02',
    LanguageCode.PORTUGUESE: 'neuralmind/bert-base-portuguese-cased',
    LanguageCode.ITALIAN: 'indigo-ai/BERTino',
    LanguageCode.TURKISH: 'dbmdz/convbert-base-turkish-mc4-uncased',
    LanguageCode.DUTCH: 'GroNLP/bert-base-dutch-cased',
    LanguageCode.UKRAINIAN: 'youscan/ukr-roberta-base',
    LanguageCode.INDONESIAN: 'indolem/indobert-base-uncased',
    LanguageCode.HINDI: 'neuralspace-reverie/indic-transformers-hi-bert',
    LanguageCode.KOREAN: 'klue/bert-base',
    LanguageCode.PERSIAN: 'HooshvareLab/albert-fa-zwnj-base-v2',
    LanguageCode.DANISH: 'Maltehb/danish-bert-botxo',
    LanguageCode.POLISH: 'dkleczek/bert-base-polish-uncased-v1',
    LanguageCode.ROMANIAN: 'Geotrend/bert-base-ro-cased',
    LanguageCode.NORWEGIAN: 'NbAiLab/nb-bert-base',
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
    LanguageCode.SINHALA: 'keshan/SinhalaBERTo',
    LanguageCode.TELUGU: 'castorini/afriberta_large',
    LanguageCode.SERBIAN: 'macedonizer/sr-roberta-base',
    LanguageCode.MACEDONIAN: 'macedonizer/mk-roberta-base',
    LanguageCode.GALICIAN: 'marcosgg/bert-base-gl-cased',
    LanguageCode.MALAY: 'malay-huggingface/bert-base-bahasa-cased',
    LanguageCode.MONGOLIAN: 'hfl/cino-base-v2',
    LanguageCode.IRISH: 'DCU-NLP/bert-base-irish-cased-v1',
    LanguageCode.MALAYALAM: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.AMHARIC: 'castorini/afriberta_large',
    LanguageCode.PUNJABI: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.KANNADA: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.NEPALI: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.JAVANESE: 'w11wo/javanese-bert-small',
    LanguageCode.WESTERN_FRISIAN: 'GroNLP/bert-base-dutch-cased-frisian',
    LanguageCode.GUARANI: 'mmaguero/multilingual-bert-gn-base-cased',
    LanguageCode.DIVEHI: 'monsoon-nlp/dv-labse',
    LanguageCode.SOMALI: 'castorini/afriberta_large',
    LanguageCode.SUNDANESE: 'w11wo/sundanese-roberta-base',
    LanguageCode.UIGHUR: 'hfl/cino-base-v2',
    LanguageCode.SANSKRIT: 'surajp/RoBERTa-hindi-guj-san',
    LanguageCode.KAZAKH: 'hfl/cino-base-v2',
    LanguageCode.BOSNIAN: 'macedonizer/ba-roberta-base',
    LanguageCode.TIBETIAN: 'hfl/cino-base-v2',
    LanguageCode.LAO: 'w11wo/lao-roberta-base',
    LanguageCode.UZBEK: 'coppercitylabs/uzbert-base-uncased',
    LanguageCode.LATIN: 'cook/cicero-similis',
    LanguageCode.BIHARI: 'ibraheemmoosa/xlmindic-base-uniscript',
    LanguageCode.TAJIK: 'muhtasham/TajBERTo',
    LanguageCode.SINDHI: 'monsoon-nlp/muril-adapted-local',
    LanguageCode.QUECHUA: 'Llamacha/QuBERTa',
    LanguageCode.ALBANIAN: 'macedonizer/al-roberta-base',
    LanguageCode.TOSK_ALBANIAN: 'macedonizer/al-roberta-base',
})


class RandomReplacementAugmenter:
    def __init__(
        self,
        target_lang: LanguageCode,
    ) -> None:
        if target_lang not in RANDOM_REPLACEMENT_MODELS:
            raise UnsupportedLanguageError(f'generation model not found for {target_lang} language')

        self._unmasker = pipeline(
            'fill-mask',
            model=RANDOM_REPLACEMENT_MODELS[target_lang],
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
                return res['sequence'].replace(self._cls_token, '').replace(self._sep_token, '').strip()

        return text
