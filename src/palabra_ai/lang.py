from dataclasses import dataclass, field

from palabra_ai.exc import ConfigurationError


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

    def __post_init__(self):
        self.code = self.code.lower()  # Always store in lowercase
        if self.registry is None:
            self.registry = _default_registry
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


AR = Language("ar", flag="🇸🇦")
AR_AE = Language("ar-ae", flag="🇦🇪")
AR_SA = Language("ar-sa", flag="🇸🇦")
AZ = Language("az", flag="🇦🇿")
BA = Language("ba", flag="🌐")  # Bashkir
BE = Language("be", flag="🇧🇾")  # Belarusian
BG = Language("bg", flag="🇧🇬")
BN = Language("bn", flag="🇧🇩")  # Bengali
BS = Language("bs", flag="🇧🇦")  # Bosnian
CA = Language("ca", flag="🌐")  # Catalan
CS = Language("cs", flag="🇨🇿")
CY = Language("cy", flag="🏴")  # Welsh
DA = Language("da", flag="🇩🇰")
DE = Language("de", flag="🇩🇪")
EL = Language("el", flag="🇬🇷")
EN = Language("en", flag="🇬🇧")
EN_AU = Language("en-au", flag="🇦🇺")
EN_CA = Language("en-ca", flag="🇨🇦")
EN_GB = Language("en-gb", flag="🇬🇧")
EN_US = Language("en-us", flag="🇺🇸")
EO = Language("eo", flag="🌐")  # Esperanto
ES = Language("es", flag="🇪🇸")
ES_MX = Language("es-mx", flag="🇲🇽")
ET = Language("et", flag="🇪🇪")  # Estonian
EU = Language("eu", flag="🌐")  # Basque
FA = Language("fa", flag="🇮🇷")  # Persian
FI = Language("fi", flag="🇫🇮")
FIL = Language("fil", flag="🇵🇭")
FR = Language("fr", flag="🇫🇷")
FR_CA = Language("fr-ca", flag="🇨🇦")
GA = Language("ga", flag="🇮🇪")  # Irish
GL = Language("gl", flag="🌐")  # Galician
HE = Language("he", flag="🇮🇱")
HI = Language("hi", flag="🇮🇳")
HR = Language("hr", flag="🇭🇷")
HU = Language("hu", flag="🇭🇺")
IA = Language("ia", flag="🌐")  # Interlingua
ID = Language("id", flag="🇮🇩")
IS = Language("is", flag="🇮🇸")  # Icelandic
IT = Language("it", flag="🇮🇹")
JA = Language("ja", flag="🇯🇵")
KK = Language("kk", flag="🇰🇿")  # Kazakh
KO = Language("ko", flag="🇰🇷")
LT = Language("lt", flag="🇱🇹")  # Lithuanian
LV = Language("lv", flag="🇱🇻")  # Latvian
MK = Language("mk", flag="🇲🇰")  # Macedonian
MN = Language("mn", flag="🇲🇳")  # Mongolian
MR = Language("mr", flag="🇮🇳")  # Marathi
MS = Language("ms", flag="🇲🇾")
MT = Language("mt", flag="🇲🇹")  # Maltese
NL = Language("nl", flag="🇳🇱")
NO = Language("no", flag="🇳🇴")
PL = Language("pl", flag="🇵🇱")
PT = Language("pt", flag="🇵🇹")
PT_BR = Language("pt-br", flag="🇧🇷")
RO = Language("ro", flag="🇷🇴")
RU = Language("ru", flag="🇷🇺")
SK = Language("sk", flag="🇸🇰")
SL = Language("sl", flag="🇸🇮")  # Slovenian
SR = Language("sr", flag="🇷🇸")  # Serbian
SV = Language("sv", flag="🇸🇪")
SW = Language("sw", flag="🇰🇪")  # Swahili
TA = Language("ta", flag="🇮🇳")
TH = Language("th", flag="🇹🇭")  # Thai
TR = Language("tr", flag="🇹🇷")
UG = Language("ug", flag="🌐")  # Uyghur
UK = Language("uk", flag="🇺🇦")
UR = Language("ur", flag="🇵🇰")  # Urdu
VI = Language("vi", flag="🇻🇳")
ZH = Language("zh", flag="🇨🇳")
ZH_HANS = Language("zh-hans", flag="🇨🇳")  # Chinese Simplified (for target)
ZH_HANT = Language("zh-hant", flag="🇹🇼")  # Chinese Traditional (for target)


# Validation for Palabra API supported languages
# Languages that support Recognition (can be used as source)
VALID_SOURCE_LANGUAGES = {
    AR, BA, BE, BG, BN, CA, CS, CY, DA, DE,
    EL, EN, EO, ES, ET, EU, FA, FI, FR, GA,
    GL, HE, HI, HR, HU, IA, ID, IT, JA, KO,
    LT, LV, MN, MR, MS, MT, NL, NO, PL, PT,
    RO, RU, SK, SL, SV, SW, TA, TH, TR, UG,
    UK, UR, VI, ZH
}

# Languages that support Translation (can be used as target)
VALID_TARGET_LANGUAGES = {
    AR, AZ, BE, BG, BS, CA, CS, CY, DA, DE,
    EL, EN, EN_AU, EN_CA, EN_GB, EN_US, ES, ES_MX, ET,
    FI, FIL, FR, FR_CA, GL, HE, HI, HR, HU,
    ID, IS, IT, JA, KK, KO, LT, LV, MK, MS,
    NL, NO, PL, PT, PT_BR, RO, RU, SK, SL, SR,
    SV, SW, TA, TR, UK, UR, VI, ZH, ZH_HANS, ZH_HANT
}

# Languages supporting auto-detection (when asr_model='alpha')
AUTO_DETECTABLE_LANGUAGES = {
    EN, UK, IT, ES, DE, PT, TR, AR, RU, PL,
    FR, ID, ZH, NL, JA, KO, FI, HU, EL, CS,
    DA, HE, HI
}


def is_valid_source_language(lang: Language) -> bool:
    """Check if language is valid for source (Recognition)"""
    return lang in VALID_SOURCE_LANGUAGES


def is_valid_target_language(lang: Language) -> bool:
    """Check if language is valid for target (Translation)"""
    return lang in VALID_TARGET_LANGUAGES


def is_auto_detectable_language(lang: Language) -> bool:
    """Check if language supports auto-detection (asr_model='alpha')"""
    return lang in AUTO_DETECTABLE_LANGUAGES
