from types import MappingProxyType

from theseus.lang_code import LanguageCode

# noinspection SpellCheckingInspection
FILL_MASK_MODELS = MappingProxyType({
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

# noinspection SpellCheckingInspection
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


BACK_TRANSLATION_MODELS = MappingProxyType({
    LanguageCode.ENGLISH: 't5-base',
})
