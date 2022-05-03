import gzip
from shutil import copyfileobj
from typing import (
    Iterable,
    Union,
)
from urllib.request import urlretrieve

import fasttext.FastText
import numpy as np
from fasttext import load_model

from theseus._paths import CACHE_DIR
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger

_SUPPORTED_LANGS = frozenset({
    LanguageCode.JAPANESE,
    LanguageCode.ARAGONESE,
    LanguageCode.KIRGHIZ,
    LanguageCode.GOAN_KONKANI,
    LanguageCode.ARMENIAN,
    LanguageCode.CEBUANO,
    LanguageCode.SERBIAN,
    LanguageCode.HINDI,
    LanguageCode.SINDHI,
    LanguageCode.PUNJABI,
    LanguageCode.NEPAL_BHASA,
    LanguageCode.ASSAMESE,
    LanguageCode.GUJARATI,
    LanguageCode.VIETNAMESE,
    LanguageCode.SANSKRIT,
    LanguageCode.YIDDISH,
    LanguageCode.YAKUT,
    LanguageCode.ROMANIAN,
    LanguageCode.IRISH,
    LanguageCode.VLAAMS,
    LanguageCode.UZBEK,
    LanguageCode.MALAYALAM,
    LanguageCode.ESTONIAN,
    LanguageCode.MAZANDERANI,
    LanguageCode.ZEEUWS,
    LanguageCode.VOLAPUK,
    LanguageCode.SOMALI,
    LanguageCode.EASTERN_MARI,
    LanguageCode.AFRIKAANS,
    LanguageCode.MALAGASY,
    LanguageCode.UIGHUR,
    LanguageCode.LATVIAN,
    LanguageCode.MACEDONIAN,
    LanguageCode.CATALAN,
    LanguageCode.ILOKO,
    LanguageCode.PORTUGUESE,
    LanguageCode.ITALIAN,
    LanguageCode.KANNADA,
    LanguageCode.LIMBURGAN,
    LanguageCode.INTERLINGUA,
    LanguageCode.OSSETIAN,
    LanguageCode.OCCITAN,
    LanguageCode.VENETIAN,
    LanguageCode.KAPAMPANGAN,
    LanguageCode.IDO,
    LanguageCode.SINHALA,
    LanguageCode.NORWEGIAN_NYNORSK,
    LanguageCode.LATIN,
    LanguageCode.MAITHILI,
    LanguageCode.FIJI_HINDI,
    LanguageCode.SOUTH_AZERBAIJANI,
    LanguageCode.BRETON,
    LanguageCode.KOREAN,
    LanguageCode.CHUVASH,
    LanguageCode.CHINESE,
    LanguageCode.FINNISH,
    LanguageCode.TIBETIAN,
    LanguageCode.POLISH,
    LanguageCode.NORTHERN_FRISIAN,
    LanguageCode.GALICIAN,
    LanguageCode.CENTRAL_KHMER,
    LanguageCode.WESTERN_MARI,
    LanguageCode.GERMAN,
    LanguageCode.MARATHI,
    LanguageCode.BULGARIAN,
    LanguageCode.URDU,
    LanguageCode.KURDISH,
    LanguageCode.BURMESE,
    LanguageCode.ORIAY,
    LanguageCode.WALLOON,
    LanguageCode.FRENCH,
    LanguageCode.CENTRAL_BIKOL,
    LanguageCode.SUNDANESE,
    LanguageCode.ARABIC,
    LanguageCode.ALBANIAN,
    LanguageCode.DANISH,
    LanguageCode.THAI,
    LanguageCode.SARDINIAN,
    LanguageCode.LITHUANIAN,
    LanguageCode.CORSICAN,
    LanguageCode.INDONESIAN,
    LanguageCode.BELARUSIAN,
    LanguageCode.MALTESE,
    LanguageCode.BOSNIAN,
    LanguageCode.CHECHEN,
    LanguageCode.CROATIAN,
    LanguageCode.MINANGKABAU,
    LanguageCode.RUSSIAN,
    LanguageCode.MANX,
    LanguageCode.SPANISH,
    LanguageCode.AZERBAIJANI,
    LanguageCode.NAHUATL,
    LanguageCode.TAJIK,
    LanguageCode.SVAHILI,
    LanguageCode.BIHARI,
    LanguageCode.HEBREW,
    LanguageCode.PASHTO,
    LanguageCode.TAGALOG,
    LanguageCode.GAELIC,
    LanguageCode.BASQUE,
    LanguageCode.GREEK,
    LanguageCode.MINGRELIAN,
    LanguageCode.BASHKIR,
    LanguageCode.TURKISH,
    LanguageCode.TOSK_ALBANIAN,
    LanguageCode.LUXEMBOURGISH,
    LanguageCode.MALAY,
    LanguageCode.ASTURIAN,
    LanguageCode.DIVEHI,
    LanguageCode.LOMBARD,
    LanguageCode.BENGALI,
    LanguageCode.EMILIANO_ROMAGNOLO,
    LanguageCode.MIRANDESE,
    LanguageCode.NEPALI,
    LanguageCode.SLOVENIAN,
    LanguageCode.BISHNUPRIYA,
    LanguageCode.WARAY,
    LanguageCode.PERSIAN,
    LanguageCode.EGYPTIAN_ARABIC,
    LanguageCode.WESTERN_FRISIAN,
    LanguageCode.ABISHIRA,
    LanguageCode.WESTERN_PANJABI,
    LanguageCode.QUECHUA,
    LanguageCode.TATAR,
    LanguageCode.SICILIAN,
    LanguageCode.KAZAKH,
    LanguageCode.PIEMONTESE,
    LanguageCode.AMHARIC,
    LanguageCode.NORTHERN_SOTHO,
    LanguageCode.ERZYA,
    LanguageCode.UKRAINIAN,
    LanguageCode.BAVARIAN,
    LanguageCode.YORUBA,
    LanguageCode.SWEDISH,
    LanguageCode.MONGOLIAN,
    LanguageCode.CZECH,
    LanguageCode.UPPER_SORBIAN,
    LanguageCode.JAVANESE,
    LanguageCode.GEORGIAN,
    LanguageCode.TELUGU,
    LanguageCode.SCOTS,
    LanguageCode.LOW_GERMAN,
    LanguageCode.SLOVAK,
    LanguageCode.WELSH,
    LanguageCode.NEAPOLITAN,
    LanguageCode.DUTCH,
    LanguageCode.HAITIAN,
    LanguageCode.NORWEGIAN,
    LanguageCode.PFAELZISCH,
    LanguageCode.TURKMEN,
    LanguageCode.ESPERANTO,
    LanguageCode.HUNGARIAN,
    LanguageCode.DIMLI,
    LanguageCode.TAMIL,
    LanguageCode.CENTRAL_KURDISH,
    LanguageCode.ENGLISH,
    LanguageCode.ICELANDIC,
    LanguageCode.ROMANSH,
})

_logger = setup_logger(__name__)

fasttext.FastText.eprint = lambda x: None


class FasttextEmbedder:
    def __init__(
        self,
        target_lang: LanguageCode,
    ) -> None:
        if target_lang not in _SUPPORTED_LANGS:
            raise UnsupportedLanguageError(f'fastText embeddings are not available for {target_lang}')

        self._target_lang = target_lang
        self._model_name = f'cc.{self._target_lang.value}.300.bin'

        self._download_model()
        self._model = load_model(str(CACHE_DIR / self._model_name))

    def __call__(
        self,
        texts: Union[str, Iterable[str]],
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        return np.asarray([self._model.get_sentence_vector(entry) for entry in texts])

    def _download_model(
        self,
    ) -> None:
        file = CACHE_DIR / self._model_name
        gz_file = CACHE_DIR / f'{file.name}.gz'

        _logger.debug(f'model path for {self._target_lang} - {file.resolve()}')

        if file.exists():
            _logger.debug(f'model for {self._target_lang} already exists, skipping download')
        else:
            url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_file.name}'
            _logger.debug(f'trying to download model for {self._target_lang} from {url}')
            urlretrieve(
                url,
                gz_file,
            )

            _logger.debug(f'model for {self._target_lang} successfully downloaded, unpacking...')

            with gzip.open(gz_file, 'rb') as gz_model:
                with open(file, 'wb') as model:
                    copyfileobj(
                        gz_model,
                        model,
                    )

            gz_file.unlink(missing_ok=True)

        _logger.debug(f'model for {self._target_lang} is ready to use')
