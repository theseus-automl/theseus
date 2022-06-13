from spacy.lang.af.stop_words import STOP_WORDS as AF_STOP_WORDS
from spacy.lang.am.stop_words import STOP_WORDS as AM_STOP_WORDS
from spacy.lang.ar.stop_words import STOP_WORDS as AR_STOP_WORDS
from spacy.lang.az.stop_words import STOP_WORDS as AZ_STOP_WORDS
from spacy.lang.bg.stop_words import STOP_WORDS as BG_STOP_WORDS
from spacy.lang.bn.stop_words import STOP_WORDS as BN_STOP_WORDS
from spacy.lang.ca.stop_words import STOP_WORDS as CA_STOP_WORDS
from spacy.lang.cs.stop_words import STOP_WORDS as CS_STOP_WORDS
from spacy.lang.da.stop_words import STOP_WORDS as DA_STOP_WORDS
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
from spacy.lang.dsb.stop_words import STOP_WORDS as DSB_STOP_WORDS
from spacy.lang.el.stop_words import STOP_WORDS as EL_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP_WORDS
from spacy.lang.et.stop_words import STOP_WORDS as ET_STOP_WORDS
from spacy.lang.eu.stop_words import STOP_WORDS as EU_STOP_WORDS
from spacy.lang.fa.stop_words import STOP_WORDS as FA_STOP_WORDS
from spacy.lang.fi.stop_words import STOP_WORDS as FI_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP_WORDS
from spacy.lang.ga.stop_words import STOP_WORDS as GA_STOP_WORDS
from spacy.lang.gu.stop_words import STOP_WORDS as GU_STOP_WORDS
from spacy.lang.he.stop_words import STOP_WORDS as HE_STOP_WORDS
from spacy.lang.hi.stop_words import STOP_WORDS as HI_STOP_WORDS
from spacy.lang.hr.stop_words import STOP_WORDS as HR_STOP_WORDS
from spacy.lang.hsb.stop_words import STOP_WORDS as HSB_STOP_WORDS
from spacy.lang.hu.stop_words import STOP_WORDS as HU_STOP_WORDS
from spacy.lang.hy.stop_words import STOP_WORDS as HY_STOP_WORDS
from spacy.lang.id.stop_words import STOP_WORDS as ID_STOP_WORDS
from spacy.lang.it.stop_words import STOP_WORDS as IT_STOP_WORDS
from spacy.lang.ja.stop_words import STOP_WORDS as JA_STOP_WORDS
from spacy.lang.kn.stop_words import STOP_WORDS as KN_STOP_WORDS
from spacy.lang.ko.stop_words import STOP_WORDS as KO_STOP_WORDS
from spacy.lang.ky.stop_words import STOP_WORDS as KY_STOP_WORDS
from spacy.lang.lb.stop_words import STOP_WORDS as LB_STOP_WORDS
from spacy.lang.lt.stop_words import STOP_WORDS as LT_STOP_WORDS
from spacy.lang.lv.stop_words import STOP_WORDS as LV_STOP_WORDS
from spacy.lang.mk.stop_words import STOP_WORDS as MK_STOP_WORDS
from spacy.lang.ml.stop_words import STOP_WORDS as ML_STOP_WORDS
from spacy.lang.mr.stop_words import STOP_WORDS as MR_STOP_WORDS
from spacy.lang.nb.stop_words import STOP_WORDS as NB_STOP_WORDS
from spacy.lang.ne.stop_words import STOP_WORDS as NE_STOP_WORDS
from spacy.lang.nl.stop_words import STOP_WORDS as NL_STOP_WORDS
from spacy.lang.pl.stop_words import STOP_WORDS as PL_STOP_WORDS
from spacy.lang.pt.stop_words import STOP_WORDS as PT_STOP_WORDS
from spacy.lang.ro.stop_words import STOP_WORDS as RO_STOP_WORDS
from spacy.lang.ru.stop_words import STOP_WORDS as RU_STOP_WORDS
from spacy.lang.sa.stop_words import STOP_WORDS as SA_STOP_WORDS
from spacy.lang.si.stop_words import STOP_WORDS as SI_STOP_WORDS
from spacy.lang.sk.stop_words import STOP_WORDS as SK_STOP_WORDS
from spacy.lang.sl.stop_words import STOP_WORDS as SL_STOP_WORDS
from spacy.lang.sq.stop_words import STOP_WORDS as SQ_STOP_WORDS
from spacy.lang.sr.stop_words import STOP_WORDS as SR_STOP_WORDS
from spacy.lang.sv.stop_words import STOP_WORDS as SV_STOP_WORDS
from spacy.lang.ta.stop_words import STOP_WORDS as TA_STOP_WORDS
from spacy.lang.te.stop_words import STOP_WORDS as TE_STOP_WORDS
from spacy.lang.th.stop_words import STOP_WORDS as TH_STOP_WORDS
from spacy.lang.tl.stop_words import STOP_WORDS as TL_STOP_WORDS
from spacy.lang.tr.stop_words import STOP_WORDS as TR_STOP_WORDS
from spacy.lang.tt.stop_words import STOP_WORDS as TT_STOP_WORDS
from spacy.lang.uk.stop_words import STOP_WORDS as UK_STOP_WORDS
from spacy.lang.ur.stop_words import STOP_WORDS as UR_STOP_WORDS
from spacy.lang.vi.stop_words import STOP_WORDS as VI_STOP_WORDS
from spacy.lang.yo.stop_words import STOP_WORDS as YO_STOP_WORDS
from spacy.lang.zh.stop_words import STOP_WORDS as ZH_STOP_WORDS

from theseus.lang_code import LanguageCode

STOP_WORDS = {
    LanguageCode.AFRIKAANS: AF_STOP_WORDS,
    LanguageCode.SLOVENIAN: SL_STOP_WORDS,
    LanguageCode.SLOVAK: SK_STOP_WORDS,
    LanguageCode.URDU: UR_STOP_WORDS,
    LanguageCode.POLISH: PL_STOP_WORDS,
    LanguageCode.VIETNAMESE: VI_STOP_WORDS,
    LanguageCode.ALBANIAN: SQ_STOP_WORDS,
    LanguageCode.SWEDISH: SV_STOP_WORDS,
    LanguageCode.IRISH: GA_STOP_WORDS,
    LanguageCode.HEBREW: HE_STOP_WORDS,
    LanguageCode.ARMENIAN: HY_STOP_WORDS,
    LanguageCode.AMHARIC: AM_STOP_WORDS,
    LanguageCode.DANISH: DA_STOP_WORDS,
    LanguageCode.MARATHI: MR_STOP_WORDS,
    LanguageCode.KIRGHIZ: KY_STOP_WORDS,
    LanguageCode.GUJARATI: GU_STOP_WORDS,
    LanguageCode.JAPANESE: JA_STOP_WORDS,
    LanguageCode.GREEK: EL_STOP_WORDS,
    LanguageCode.LATVIAN: LV_STOP_WORDS,
    LanguageCode.LUXEMBOURGISH: LB_STOP_WORDS,
    LanguageCode.ITALIAN: IT_STOP_WORDS,
    LanguageCode.CATALAN: CA_STOP_WORDS,
    LanguageCode.ICELANDIC: set(
        """
        afhverju
        aftan
        aftur
        afþví
        aldrei
        allir
        allt
        alveg
        annað
        annars
        bara
        dag
        eða
        eftir
        eiga
        einhver
        einhverjir
        einhvers
        eins
        einu
        eitthvað
        ekkert
        ekki
        ennþá
        eru
        fara
        fer
        finna
        fjöldi
        fólk
        framan
        frá
        frekar
        fyrir
        gegnum
        geta
        getur
        gmg
        gott
        hann
        hafa
        hef
        hefur
        heyra
        hér
        hérna
        hjá
        hún
        hvað
        hvar
        hver
        hverjir
        hverjum
        hvernig
        hvor
        hvort
        hægt
        img
        inn
        kannski
        koma
        líka
        lol
        maður
        mátt
        mér
        með
        mega
        meira
        mig
        mikið
        minna
        minni
        missa
        mjög
        nei
        niður
        núna
        oft
        okkar
        okkur
        póst
        póstur
        rofl
        saman
        sem
        sér
        sig
        sinni
        síðan
        sjá
        smá
        smátt
        spurja
        spyrja
        staðar
        stórt
        svo
        svona
        sælir
        sæll
        taka
        takk
        til
        tilvitnun
        titlar
        upp
        var
        vel
        velkomin
        velkominn
        vera
        verður
        verið
        vel
        við
        vil
        vilja
        vill
        vita
        væri
        yfir
        ykkar
        það
        þakka
        þakkir
        þannig
        það
        þar
        þarf
        þau
        þeim
        þeir
        þeirra
        þeirra
        þegar
        þess
        þessa
        þessi
        þessu
        þessum
        þetta
        þér
        þið
        þinn
        þitt
        þín
        þráð
        þráður
        því
        þær
        ætti
        """.split()
    ),
    LanguageCode.CZECH: CS_STOP_WORDS,
    LanguageCode.TELUGU: TE_STOP_WORDS,
    LanguageCode.RUSSIAN: RU_STOP_WORDS,
    LanguageCode.TAGALOG: TL_STOP_WORDS,
    LanguageCode.ROMANIAN: RO_STOP_WORDS,
    LanguageCode.UPPER_SORBIAN: HSB_STOP_WORDS,
    LanguageCode.YORUBA: YO_STOP_WORDS,
    LanguageCode.SANSKRIT: SA_STOP_WORDS,
    LanguageCode.PORTUGUESE: PT_STOP_WORDS,
    LanguageCode.CHINESE: ZH_STOP_WORDS,
    LanguageCode.UKRAINIAN: UK_STOP_WORDS,
    LanguageCode.SERBIAN: SR_STOP_WORDS,
    LanguageCode.SINHALA: SI_STOP_WORDS,
    LanguageCode.MALAYALAM: ML_STOP_WORDS,
    LanguageCode.MACEDONIAN: MK_STOP_WORDS,
    LanguageCode.KANNADA: KN_STOP_WORDS,
    LanguageCode.ARABIC: AR_STOP_WORDS,
    LanguageCode.CROATIAN: HR_STOP_WORDS,
    LanguageCode.HUNGARIAN: HU_STOP_WORDS,
    LanguageCode.DUTCH: NL_STOP_WORDS,
    LanguageCode.BULGARIAN: BG_STOP_WORDS,
    LanguageCode.BENGALI: BN_STOP_WORDS,
    LanguageCode.NEPALI: NE_STOP_WORDS,
    LanguageCode.NORWEGIAN: NB_STOP_WORDS,
    LanguageCode.HINDI: HI_STOP_WORDS,
    LanguageCode.GERMAN: DE_STOP_WORDS,
    LanguageCode.AZERBAIJANI: AZ_STOP_WORDS,
    LanguageCode.KOREAN: KO_STOP_WORDS,
    LanguageCode.FINNISH: FI_STOP_WORDS,
    LanguageCode.INDONESIAN: ID_STOP_WORDS,
    LanguageCode.FRENCH: FR_STOP_WORDS,
    LanguageCode.SPANISH: ES_STOP_WORDS,
    LanguageCode.ESTONIAN: ET_STOP_WORDS,
    LanguageCode.ENGLISH: EN_STOP_WORDS,
    LanguageCode.PERSIAN: FA_STOP_WORDS,
    LanguageCode.LITHUANIAN: LT_STOP_WORDS,
    LanguageCode.BASQUE: EU_STOP_WORDS,
    LanguageCode.TATAR: TT_STOP_WORDS,
    LanguageCode.LOWER_SORBIAN: DSB_STOP_WORDS,
    LanguageCode.TAMIL: TA_STOP_WORDS,
    LanguageCode.THAI: TH_STOP_WORDS,
    LanguageCode.TURKISH: TR_STOP_WORDS,
}
