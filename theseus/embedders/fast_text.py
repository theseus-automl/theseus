import re
import warnings
from typing import (
    Any,
    Iterable,
    Union,
)
from urllib.request import urlretrieve

import numpy as np
from compress_fasttext.models import CompressedFastTextKeyedVectors
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from theseus._paths import CACHE_DIR
from theseus.exceptions import UnsupportedLanguageError
from theseus.lang_code import LanguageCode
from theseus.log import setup_logger
from theseus.stop_words import STOP_WORDS

warnings.simplefilter(
    'ignore',
    RuntimeWarning,
)

FT_SUPPORTED_LANGS = frozenset((
    LanguageCode.AFRIKAANS,
    LanguageCode.TOSK_ALBANIAN,
    LanguageCode.AMHARIC,
    LanguageCode.ARAGONESE,
    LanguageCode.ARABIC,
    LanguageCode.EGYPTIAN_ARABIC,
    LanguageCode.ASSAMESE,
    LanguageCode.ASTURIAN,
    LanguageCode.AZERBAIJANI,
    LanguageCode.SOUTH_AZERBAIJANI,
    LanguageCode.BASHKIR,
    LanguageCode.BAVARIAN,
    LanguageCode.BELARUSIAN,
    LanguageCode.CENTRAL_BIKOL,
    LanguageCode.BULGARIAN,
    LanguageCode.BIHARI,
    LanguageCode.BENGALI,
    LanguageCode.TIBETIAN,
    LanguageCode.BISHNUPRIYA,
    LanguageCode.BRETON,
    LanguageCode.BOSNIAN,
    LanguageCode.CATALAN,
    LanguageCode.CHECHEN,
    LanguageCode.CEBUANO,
    LanguageCode.CENTRAL_KURDISH,
    LanguageCode.CORSICAN,
    LanguageCode.CZECH,
    LanguageCode.CHUVASH,
    LanguageCode.WELSH,
    LanguageCode.DANISH,
    LanguageCode.GERMAN,
    LanguageCode.DIMLI,
    LanguageCode.DIVEHI,
    LanguageCode.GREEK,
    LanguageCode.EMILIANO_ROMAGNOLO,
    LanguageCode.ENGLISH,
    LanguageCode.ESPERANTO,
    LanguageCode.SPANISH,
    LanguageCode.ESTONIAN,
    LanguageCode.BASQUE,
    LanguageCode.PERSIAN,
    LanguageCode.FINNISH,
    LanguageCode.FRENCH,
    LanguageCode.NORTHERN_FRISIAN,
    LanguageCode.WESTERN_FRISIAN,
    LanguageCode.IRISH,
    LanguageCode.GAELIC,
    LanguageCode.GALICIAN,
    LanguageCode.GOAN_KONKANI,
    LanguageCode.GUJARATI,
    LanguageCode.MANX,
    LanguageCode.HEBREW,
    LanguageCode.HINDI,
    LanguageCode.FIJI_HINDI,
    LanguageCode.CROATIAN,
    LanguageCode.UPPER_SORBIAN,
    LanguageCode.HAITIAN,
    LanguageCode.HUNGARIAN,
    LanguageCode.ARMENIAN,
    LanguageCode.INTERLINGUA,
    LanguageCode.INDONESIAN,
    LanguageCode.ILOKO,
    LanguageCode.IDO,
    LanguageCode.ICELANDIC,
    LanguageCode.ITALIAN,
    LanguageCode.JAPANESE,
    LanguageCode.JAVANESE,
    LanguageCode.GEORGIAN,
    LanguageCode.KAZAKH,
    LanguageCode.CENTRAL_KHMER,
    LanguageCode.KANNADA,
    LanguageCode.KOREAN,
    LanguageCode.KURDISH,
    LanguageCode.KIRGHIZ,
    LanguageCode.LATIN,
    LanguageCode.LUXEMBOURGISH,
    LanguageCode.LIMBURGAN,
    LanguageCode.LOMBARD,
    LanguageCode.LITHUANIAN,
    LanguageCode.LATVIAN,
    LanguageCode.MAITHILI,
    LanguageCode.MALAGASY,
    LanguageCode.EASTERN_MARI,
    LanguageCode.MINANGKABAU,
    LanguageCode.MACEDONIAN,
    LanguageCode.MALAYALAM,
    LanguageCode.MONGOLIAN,
    LanguageCode.MARATHI,
    LanguageCode.WESTERN_MARI,
    LanguageCode.MALAY,
    LanguageCode.MALTESE,
    LanguageCode.MIRANDESE,
    LanguageCode.BURMESE,
    LanguageCode.ERZYA,
    LanguageCode.MAZANDERANI,
    LanguageCode.NAHUATL,
    LanguageCode.NEAPOLITAN,
    LanguageCode.LOW_GERMAN,
    LanguageCode.NEPALI,
    LanguageCode.NEPAL_BHASA,
    LanguageCode.DUTCH,
    LanguageCode.RUSSIAN,
))

_logger = setup_logger(__name__)


class FasttextEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_lang: LanguageCode,
    ) -> None:
        if target_lang not in FT_SUPPORTED_LANGS:
            raise UnsupportedLanguageError(f'fastText embeddings are not available for {target_lang}')

        self.target_lang = target_lang
        self._model_name = f'fasttext_{self.target_lang.value}_mini'

        self._download_model()
        self._model = CompressedFastTextKeyedVectors.load(str(CACHE_DIR / self._model_name))
        self._token_regex = re.compile(r'(?u)\b\w\w+\b')
        self._stop_words = STOP_WORDS.get(
            self.target_lang,
            set(),
        )

    def fit(
        self,
        texts: Union[str, Iterable[str]],
        y: Any = None,
    ) -> 'FasttextEmbedder':
        return self

    def transform(
        self,
        texts: Union[str, Iterable[str]],
        y: Any = None,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for entry in texts:
            entry = self._preprocess(entry)

            if entry:
                embeddings.append(np.nan_to_num(self._model.get_sentence_vector(entry)))
            else:
                embeddings.append(np.zeros(300))

        return np.asarray(embeddings)

    def _download_model(
        self,
    ) -> None:
        file = CACHE_DIR / self._model_name
        _logger.debug(f'model path for {self.target_lang} - {file.resolve()}')

        if file.exists():
            _logger.debug(f'model for {self.target_lang} already exists, skipping download')
        else:
            if self.target_lang == LanguageCode.RUSSIAN:
                url = 'https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/geowac_tokens_sg_300_5_2020-100K-20K-100.bin'
            else:
                url = f'https://zenodo.org/record/4905385/files/fasttext-{self.target_lang.value}-mini?download=1'

            _logger.debug(f'trying to download model for {self.target_lang} from {url}')

            urlretrieve(
                url,
                file,
            )

            _logger.debug(f'model for {self.target_lang} successfully downloaded')

        _logger.debug(f'model for {self.target_lang} is ready to use')

    def _preprocess(
        self,
        text: str,
    ) -> str:
        text = text.strip().lower()
        words = []

        for word in self._token_regex.findall(text):
            if word not in self._stop_words:
                words.append(word)

        return ' '.join(words)
