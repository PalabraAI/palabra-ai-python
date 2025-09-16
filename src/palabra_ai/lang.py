from dataclasses import dataclass, field
from typing import Any

from palabra_ai.exc import ConfigurationError

# Sentinel value to distinguish between "not provided" and "explicitly None"
_UNSET = object()


@dataclass
class LanguageRegistry:
    by_code: dict[str, "Language"] = field(
        default_factory=dict, repr=False, compare=False
    )
    all_languages: set["Language"] = field(
        default_factory=set, repr=False, compare=False
    )

    def register(self, language: "Language"):
        self.by_code[language.code] = language
        self.all_languages.add(language)

    def get_by_bcp47(self, code: str) -> "Language | None":
        if result := self.by_code.get(code.lower()):
            return result
        raise ConfigurationError(f"Language with BCP47 code '{code}' not found.")

    def get_or_create(self, code: str) -> "Language":
        """Get existing language or create new one dynamically"""
        code_lower = code.lower()
        try:
            return self.get_by_bcp47(code_lower)
        except ConfigurationError:
            # Create new language dynamically
            return Language(code_lower, registry=self)


_default_registry = LanguageRegistry()


@dataclass
class Language:
    code: str
    registry: LanguageRegistry = field(default=None, repr=False, compare=False)
    flag: str = "🌐❓"
    source_code: str | None | Any = field(
        default=_UNSET
    )  # Code to use for source (recognition)
    target_code: str | None | Any = field(
        default=_UNSET
    )  # Code to use for target (translation)

    def __post_init__(self):
        self.code = self.code.lower()  # Always store in lowercase
        if self.registry is None:
            self.registry = _default_registry

        # Set default mappings if not provided (only if not explicitly set)
        if self.source_code is _UNSET:
            self.source_code = self.code
        if self.target_code is _UNSET:
            self.target_code = self.code

        self.registry.register(self)

    @property
    def bcp47(self) -> str:
        return self.code

    @classmethod
    def get_by_bcp47(
        cls, code: str, registry: LanguageRegistry | None = None
    ) -> "Language | None":
        if registry is None:
            registry = _default_registry
        return registry.get_by_bcp47(code)

    @classmethod
    def get_or_create(
        cls, code: str, registry: LanguageRegistry | None = None
    ) -> "Language":
        """Get existing language or create new one dynamically"""
        if registry is None:
            registry = _default_registry
        return registry.get_or_create(code)

    def __hash__(self):
        return hash(self.code)

    def __str__(self):
        return self.bcp47

    def __repr__(self):
        return f"{self.flag}{str(self)}"

    def __eq__(self, other):
        if isinstance(other, Language):
            return self.code == other.code
        elif isinstance(other, str):
            # Check if string exists as a language code in registry
            if other.lower() in self.registry.by_code:
                return self.code == other.lower()
            raise TypeError(
                f"Cannot compare Language with unknown language code: {other}"
            )
        else:
            raise TypeError(f"Cannot compare Language with {type(other).__name__}")


# Languages with custom mapping for source/target codes
AR = Language("ar", flag="🇸🇦", source_code="ar", target_code="ar")
AR_AE = Language("ar-ae", flag="🇦🇪", source_code="ar", target_code="ar-ae")
AR_SA = Language("ar-sa", flag="🇸🇦", source_code="ar", target_code="ar-sa")
AZ = Language("az", flag="🇦🇿", source_code=None, target_code="az")  # Target only
BA = Language(
    "ba", flag="🌐", source_code="ba", target_code=None
)  # Bashkir - Source only
BE = Language("be", flag="🇧🇾", source_code="be", target_code="be")  # Belarusian
BG = Language("bg", flag="🇧🇬", source_code="bg", target_code="bg")
BN = Language(
    "bn", flag="🇧🇩", source_code="bn", target_code=None
)  # Bengali - Source only
BS = Language(
    "bs", flag="🇧🇦", source_code=None, target_code="bs"
)  # Bosnian - Target only
CA = Language("ca", flag="🌐", source_code="ca", target_code="ca")  # Catalan
CS = Language("cs", flag="🇨🇿", source_code="cs", target_code="cs")
CY = Language("cy", flag="🏴", source_code="cy", target_code="cy")  # Welsh
DA = Language("da", flag="🇩🇰", source_code="da", target_code="da")
DE = Language("de", flag="🇩🇪", source_code="de", target_code="de")
EL = Language("el", flag="🇬🇷", source_code="el", target_code="el")
EN = Language(
    "en", flag="🇬🇧", source_code="en", target_code="en-us"
)  # Base English -> smart mapping
EN_AU = Language("en-au", flag="🇦🇺", source_code="en", target_code="en-au")
EN_CA = Language("en-ca", flag="🇨🇦", source_code="en", target_code="en-ca")
EN_GB = Language("en-gb", flag="🇬🇧", source_code="en", target_code="en-gb")
EN_US = Language("en-us", flag="🇺🇸", source_code="en", target_code="en-us")
EO = Language(
    "eo", flag="🌐", source_code="eo", target_code=None
)  # Esperanto - Source only
ES = Language("es", flag="🇪🇸", source_code="es", target_code="es")
ES_MX = Language("es-mx", flag="🇲🇽", source_code="es", target_code="es-mx")
ET = Language("et", flag="🇪🇪", source_code="et", target_code="et")  # Estonian
EU = Language(
    "eu", flag="🌐", source_code="eu", target_code=None
)  # Basque - Source only
FA = Language(
    "fa", flag="🇮🇷", source_code="fa", target_code=None
)  # Persian - Source only
FI = Language("fi", flag="🇫🇮", source_code="fi", target_code="fi")
FIL = Language(
    "fil", flag="🇵🇭", source_code=None, target_code="fil"
)  # Filipino - Target only
FR = Language("fr", flag="🇫🇷", source_code="fr", target_code="fr")
FR_CA = Language("fr-ca", flag="🇨🇦", source_code="fr", target_code="fr-ca")
GA = Language(
    "ga", flag="🇮🇪", source_code="ga", target_code=None
)  # Irish - Source only
GL = Language("gl", flag="🌐", source_code="gl", target_code="gl")  # Galician
HE = Language("he", flag="🇮🇱", source_code="he", target_code="he")
HI = Language("hi", flag="🇮🇳", source_code="hi", target_code="hi")
HR = Language("hr", flag="🇭🇷", source_code="hr", target_code="hr")
HU = Language("hu", flag="🇭🇺", source_code="hu", target_code="hu")
IA = Language(
    "ia", flag="🌐", source_code="ia", target_code=None
)  # Interlingua - Source only
ID = Language("id", flag="🇮🇩", source_code="id", target_code="id")
IS = Language(
    "is", flag="🇮🇸", source_code=None, target_code="is"
)  # Icelandic - Target only
IT = Language("it", flag="🇮🇹", source_code="it", target_code="it")
JA = Language("ja", flag="🇯🇵", source_code="ja", target_code="ja")
KK = Language(
    "kk", flag="🇰🇿", source_code=None, target_code="kk"
)  # Kazakh - Target only
KO = Language("ko", flag="🇰🇷", source_code="ko", target_code="ko")
LT = Language("lt", flag="🇱🇹", source_code="lt", target_code="lt")  # Lithuanian
LV = Language("lv", flag="🇱🇻", source_code="lv", target_code="lv")  # Latvian
MK = Language(
    "mk", flag="🇲🇰", source_code=None, target_code="mk"
)  # Macedonian - Target only
MN = Language(
    "mn", flag="🇲🇳", source_code="mn", target_code=None
)  # Mongolian - Source only
MR = Language(
    "mr", flag="🇮🇳", source_code="mr", target_code=None
)  # Marathi - Source only
MS = Language("ms", flag="🇲🇾", source_code="ms", target_code="ms")
MT = Language(
    "mt", flag="🇲🇹", source_code="mt", target_code=None
)  # Maltese - Source only
NL = Language("nl", flag="🇳🇱", source_code="nl", target_code="nl")
NO = Language("no", flag="🇳🇴", source_code="no", target_code="no")
PL = Language("pl", flag="🇵🇱", source_code="pl", target_code="pl")
PT = Language("pt", flag="🇵🇹", source_code="pt", target_code="pt")
PT_BR = Language("pt-br", flag="🇧🇷", source_code="pt", target_code="pt-br")
RO = Language("ro", flag="🇷🇴", source_code="ro", target_code="ro")
RU = Language("ru", flag="🇷🇺", source_code="ru", target_code="ru")
SK = Language("sk", flag="🇸🇰", source_code="sk", target_code="sk")
SL = Language("sl", flag="🇸🇮", source_code="sl", target_code="sl")  # Slovenian
SR = Language(
    "sr", flag="🇷🇸", source_code=None, target_code="sr"
)  # Serbian - Target only
SV = Language("sv", flag="🇸🇪", source_code="sv", target_code="sv")
SW = Language("sw", flag="🇰🇪", source_code="sw", target_code="sw")  # Swahili
TA = Language("ta", flag="🇮🇳", source_code="ta", target_code="ta")
TH = Language("th", flag="🇹🇭", source_code="th", target_code=None)  # Thai - Source only
TR = Language("tr", flag="🇹🇷", source_code="tr", target_code="tr")
UG = Language(
    "ug", flag="🌐", source_code="ug", target_code=None
)  # Uyghur - Source only
UK = Language("uk", flag="🇺🇦", source_code="uk", target_code="uk")
UR = Language("ur", flag="🇵🇰", source_code="ur", target_code="ur")  # Urdu
VI = Language("vi", flag="🇻🇳", source_code="vi", target_code="vi")
ZH = Language(
    "zh", flag="🇨🇳", source_code="zh", target_code="zh-hans"
)  # Base Chinese -> smart mapping
ZH_HANS = Language(
    "zh-hans", flag="🇨🇳", source_code="zh", target_code="zh-hans"
)  # Chinese Simplified
ZH_HANT = Language(
    "zh-hant", flag="🇹🇼", source_code="zh", target_code="zh-hant"
)  # Chinese Traditional


# Validation for Palabra API supported languages
# Languages that support Recognition (can be used as source)
VALID_SOURCE_LANGUAGES = {
    AR,
    BA,
    BE,
    BG,
    BN,
    CA,
    CS,
    CY,
    DA,
    DE,
    EL,
    EN,
    EN_AU,  # English variants can be used as source (map to "en")
    EN_CA,
    EN_GB,
    EN_US,
    EO,
    ES,
    ES_MX,  # Spanish variants can be used as source (map to "es")
    ET,
    EU,
    FA,
    FI,
    FR,
    FR_CA,  # French variants can be used as source (map to "fr")
    GA,
    GL,
    HE,
    HI,
    HR,
    HU,
    IA,
    ID,
    IT,
    JA,
    KO,
    LT,
    LV,
    MN,
    MR,
    MS,
    MT,
    NL,
    NO,
    PL,
    PT,
    PT_BR,  # Portuguese variants can be used as source (map to "pt")
    RO,
    RU,
    SK,
    SL,
    SV,
    SW,
    TA,
    TH,
    TR,
    UG,
    UK,
    UR,
    VI,
    ZH,
    ZH_HANS,  # Chinese variants can be used as source (map to "zh")
    ZH_HANT,
}

# Languages that support Translation (can be used as target)
# Note: EN and ZH are handled via smart mapping (EN -> EN_US, ZH -> ZH_HANS)
VALID_TARGET_LANGUAGES = {
    AR,
    AZ,
    BE,
    BG,
    BS,
    CA,
    CS,
    CY,
    DA,
    DE,
    EL,
    EN,  # Kept for backward compatibility - will map to EN_US via smart mapping
    EN_AU,
    EN_CA,
    EN_GB,
    EN_US,
    ES,
    ES_MX,
    ET,
    FI,
    FIL,
    FR,
    FR_CA,
    GL,
    HE,
    HI,
    HR,
    HU,
    ID,
    IS,
    IT,
    JA,
    KK,
    KO,
    LT,
    LV,
    MK,
    MS,
    NL,
    NO,
    PL,
    PT,
    PT_BR,
    RO,
    RU,
    SK,
    SL,
    SR,
    SV,
    SW,
    TA,
    TR,
    UK,
    UR,
    VI,
    ZH,  # Kept for backward compatibility - will map to ZH_HANS via smart mapping
    ZH_HANS,
    ZH_HANT,
}

# Languages supporting auto-detection (when asr_model='alpha')
AUTO_DETECTABLE_LANGUAGES = {
    EN,
    UK,
    IT,
    ES,
    DE,
    PT,
    TR,
    AR,
    RU,
    PL,
    FR,
    ID,
    ZH,
    NL,
    JA,
    KO,
    FI,
    HU,
    EL,
    CS,
    DA,
    HE,
    HI,
}


def is_valid_source_language(lang: Language) -> bool:
    """Check if language is valid for source (Recognition)"""
    return lang.source_code is not None and lang in VALID_SOURCE_LANGUAGES


def is_valid_target_language(lang: Language) -> bool:
    """Check if language is valid for target (Translation)"""
    return lang.target_code is not None and lang in VALID_TARGET_LANGUAGES


def is_auto_detectable_language(lang: Language) -> bool:
    """Check if language supports auto-detection (asr_model='alpha')"""
    return lang in AUTO_DETECTABLE_LANGUAGES
