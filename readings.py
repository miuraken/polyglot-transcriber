"""
Reading (pronunciation guide) generators for transcribed text.

Each supported language maps to one or more reading styles:
  - "romaji": Latin-script transliteration (IAST for Indic scripts,
              Revised Romanization with pronunciation rules for Korean)
  - "kana":   Japanese katakana approximation of the pronunciation (Korean only)
"""
import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

try:
    import koroman
    from koroman import core as _koroman_core
except ImportError:  # Korean readings become unavailable, everything else still works
    koroman = None
    _koroman_core = None


# --- Indic scripts -> IAST ---------------------------------------------------
# IAST is designed for Indic (Brahmic) scripts, so only these are supported.
IAST_SOURCE_SCRIPTS = {
    "hi": sanscript.DEVANAGARI,  # Hindi
    "mr": sanscript.DEVANAGARI,  # Marathi
    "ne": sanscript.DEVANAGARI,  # Nepali
    "sa": sanscript.DEVANAGARI,  # Sanskrit
    "bn": sanscript.BENGALI,     # Bengali
    "gu": sanscript.GUJARATI,    # Gujarati
    "pa": sanscript.GURMUKHI,    # Punjabi
    "ta": sanscript.TAMIL,       # Tamil
    "te": sanscript.TELUGU,      # Telugu
    "kn": sanscript.KANNADA,     # Kannada
    "ml": sanscript.MALAYALAM,   # Malayalam
}


def _iast(lang):
    return lambda text: transliterate(text, IAST_SOURCE_SCRIPTS[lang], sanscript.IAST)


# --- Korean -> Revised Romanization -----------------------------------------
def korean_romaji(text):
    """Revised Romanization with pronunciation rules (e.g. 감사합니다 -> gamsahamnida)."""
    return koroman.romanize(text)


# --- Korean -> Katakana -------------------------------------------------------
# Conjoining jamo code point ranges (output of koroman's jamo decomposition)
_CHO_RANGE = range(0x1100, 0x1113)
_JUNG_RANGE = range(0x1161, 0x1176)
_JONG_RANGE = range(0x11A8, 0x11C3)

# Initial consonant -> (unvoiced row, voiced row, kind)
#   kind: "lenis" voices between voiced sounds, "tense" gets a preceding small tsu
_CHO = {
    "ᄀ": ("k", "g", "lenis"), "ᄁ": ("k", "k", "tense"), "ᄏ": ("k", "k", "asp"),
    "ᄂ": ("n", "n", "son"),
    "ᄃ": ("t", "d", "lenis"), "ᄄ": ("t", "t", "tense"), "ᄐ": ("t", "t", "asp"),
    "ᄅ": ("r", "r", "son"),
    "ᄆ": ("m", "m", "son"),
    "ᄇ": ("p", "b", "lenis"), "ᄈ": ("p", "p", "tense"), "ᄑ": ("p", "p", "asp"),
    "ᄉ": ("s", "s", "asp"), "ᄊ": ("s", "s", "tense"),
    "ᄋ": ("", "", "son"),
    "ᄌ": ("ch", "j", "lenis"), "ᄍ": ("ch", "ch", "tense"), "ᄎ": ("ch", "ch", "asp"),
    "ᄒ": ("h", "h", "asp"),
}

# Vowel -> (glide, base vowel). glide: "" / "y" / "w"
_JUNG = {
    "ᅡ": ("", "a"), "ᅢ": ("", "e"), "ᅣ": ("y", "a"), "ᅤ": ("y", "e"),
    "ᅥ": ("", "o"), "ᅦ": ("", "e"), "ᅧ": ("y", "o"), "ᅨ": ("y", "e"),
    "ᅩ": ("", "o"), "ᅪ": ("w", "a"), "ᅫ": ("w", "e"), "ᅬ": ("w", "e"),
    "ᅭ": ("y", "o"), "ᅮ": ("", "u"), "ᅯ": ("w", "o"), "ᅰ": ("w", "e"),
    "ᅱ": ("w", "i"), "ᅲ": ("y", "u"), "ᅳ": ("", "u"), "ᅴ": ("", "ui"), "ᅵ": ("", "i"),
}

# Final consonant -> representative katakana (small kana as used in Korean textbooks)
_JONG = {}
for _chars, _kana in (
    ("ᆨᆩᆿᆪᆰ", "ㇰ"), ("ᆫᆬᆭ", "ン"), ("ᆮᆺᆻᆽᆾᇀᇂ", "ッ"),
    ("ᆯᆲᆳᆴᆶ", "ㇽ"), ("ᆷᆱ", "ㇺ"), ("ᆸᇁᆹᆵ", "ㇷ゚"), ("ᆼ", "ン"),
):
    for _c in _chars:
        _JONG[_c] = _kana
_VOICED_JONG = set("ᆫᆬᆭᆯᆲᆳᆴᆶᆷᆱᆼ")  # sonorant finals keep the next lenis consonant voiced

# Liaison (연음): final consonant + silent initial ㅇ -> the consonant moves to the next syllable.
# koroman's romaji makes this implicit (saram+i -> sarami), but kana needs it explicitly.
# Value: (remaining final or "", new initial or "")
_LIAISON = {
    "ᆨ": ("", "ᄀ"), "ᆩ": ("", "ᄁ"), "ᆫ": ("", "ᄂ"), "ᆮ": ("", "ᄃ"),
    "ᆯ": ("", "ᄅ"), "ᆷ": ("", "ᄆ"), "ᆸ": ("", "ᄇ"), "ᆺ": ("", "ᄉ"),
    "ᆻ": ("", "ᄊ"), "ᆽ": ("", "ᄌ"), "ᆾ": ("", "ᄎ"), "ᆿ": ("", "ᄏ"),
    "ᇀ": ("", "ᄐ"), "ᇁ": ("", "ᄑ"), "ᇂ": ("", ""),
    "ᆪ": ("ᆨ", "ᄊ"), "ᆬ": ("ᆫ", "ᄌ"), "ᆭ": ("", "ᄂ"), "ᆰ": ("ᆯ", "ᄀ"),
    "ᆱ": ("ᆯ", "ᄆ"), "ᆲ": ("ᆯ", "ᄇ"), "ᆳ": ("ᆯ", "ᄊ"), "ᆴ": ("ᆯ", "ᄐ"),
    "ᆵ": ("ᆯ", "ᄑ"), "ᆶ": ("", "ᄅ"), "ᆹ": ("ᆸ", "ᄊ"),
}
_LIAISON_RE = re.compile("([" + "".join(_LIAISON) + "])ᄋ(?=[\u1161-\u1175])")


def _apply_liaison(jamo):
    def repl(m):
        keep, new_cho = _LIAISON[m.group(1)]
        return keep + (new_cho or "ᄋ")
    return _LIAISON_RE.sub(repl, jamo)

_KANA = {
    "":   dict(a="ア", i="イ", u="ウ", e="エ", o="オ"),
    "k":  dict(a="カ", i="キ", u="ク", e="ケ", o="コ"),
    "g":  dict(a="ガ", i="ギ", u="グ", e="ゲ", o="ゴ"),
    "s":  dict(a="サ", i="シ", u="ス", e="セ", o="ソ"),
    "ch": dict(a="チャ", i="チ", u="チュ", e="チェ", o="チョ"),
    "j":  dict(a="ジャ", i="ジ", u="ジュ", e="ジェ", o="ジョ"),
    "t":  dict(a="タ", i="ティ", u="トゥ", e="テ", o="ト"),
    "d":  dict(a="ダ", i="ディ", u="ドゥ", e="デ", o="ド"),
    "n":  dict(a="ナ", i="ニ", u="ヌ", e="ネ", o="ノ"),
    "h":  dict(a="ハ", i="ヒ", u="フ", e="ヘ", o="ホ"),
    "p":  dict(a="パ", i="ピ", u="プ", e="ペ", o="ポ"),
    "b":  dict(a="バ", i="ビ", u="ブ", e="ベ", o="ボ"),
    "m":  dict(a="マ", i="ミ", u="ム", e="メ", o="モ"),
    "r":  dict(a="ラ", i="リ", u="ル", e="レ", o="ロ"),
}
_SMALL_Y = dict(a="ャ", u="ュ", o="ョ", e="ェ")
_SMALL_W = dict(a="ァ", i="ィ", e="ェ", o="ォ")


def _syllable_kana(row, glide, vowel):
    """Builds katakana for one syllable from consonant row + vowel."""
    if vowel == "ui":  # ㅢ: 의 -> ウィ, 희 -> ヒ
        return "ウィ" if row == "" else _KANA[row]["i"]
    if glide == "y":
        if row == "":
            return {"a": "ヤ", "u": "ユ", "o": "ヨ", "e": "イェ"}[vowel]
        if row in ("ch", "j"):  # already palatal: 쟈 -> ジャ
            return _KANA[row][vowel]
        return _KANA[row]["i"] + _SMALL_Y[vowel]
    if glide == "w":
        if row == "":
            return {"a": "ワ", "i": "ウィ", "e": "ウェ", "o": "ウォ"}[vowel]
        return _KANA[row]["u"] + _SMALL_W[vowel]
    return _KANA[row][vowel]


def korean_kana(text):
    """
    Approximates Korean pronunciation in katakana (e.g. 감사합니다 -> カㇺサハㇺニダ).
    Uses koroman's pronunciation rules (liaison, nasalization, etc.), then maps
    each pronounced syllable to katakana, voicing lenis consonants between voiced sounds.
    """
    c = _koroman_core
    jamo = c.apply_pronunciation_rules(c.split_hangul_to_jamos(c.apply_n_insertion_triggers(text)))
    jamo = jamo.replace("ll", "ᆯᄅ")  # koroman marks ㄹ+ㄹ with a Latin "ll" placeholder
    jamo = _apply_liaison(jamo)

    out = []
    prev_voiced_end = False  # did the previous syllable in this word end in a voiced sound?
    prev_open = False        # did the previous syllable in this word end in a vowel?
    i, n = 0, len(jamo)
    while i < n:
        ch = jamo[i]
        code = ord(ch)
        if code in _CHO_RANGE or code in _JUNG_RANGE:
            cho = "ᄋ"
            if code in _CHO_RANGE:
                cho = ch
                i += 1
            if i >= n or ord(jamo[i]) not in _JUNG_RANGE:  # stray initial consonant
                out.append(ch)
                prev_voiced_end = prev_open = False
                continue
            jung = jamo[i]
            i += 1
            jong = ""
            if i < n and ord(jamo[i]) in _JONG_RANGE and not (i + 1 < n and ord(jamo[i + 1]) in _JUNG_RANGE):
                jong = jamo[i]
                i += 1

            unvoiced, voiced, kind = _CHO.get(cho, ("", "", "son"))
            row = voiced if (kind == "lenis" and prev_voiced_end) else unvoiced
            if kind == "tense" and prev_open:
                out.append("ッ")
            glide, vowel = _JUNG[jung]
            out.append(_syllable_kana(row, glide, vowel))
            if jong:
                out.append(_JONG.get(jong, ""))
            prev_voiced_end = (not jong) or jong in _VOICED_JONG
            prev_open = not jong
        elif code in _JONG_RANGE:  # stray final consonant
            out.append(_JONG.get(ch, ""))
            prev_voiced_end = prev_open = False
            i += 1
        else:  # spaces / punctuation / other scripts reset word context
            out.append(ch)
            prev_voiced_end = prev_open = False
            i += 1
    return "".join(out)


# --- Registry -------------------------------------------------------------------
READING_STYLES = ("romaji", "kana", "both", "none")


def _build_registry():
    reg = {lang: {"romaji": _iast(lang)} for lang in IAST_SOURCE_SCRIPTS}
    if koroman is not None:
        reg["ko"] = {"romaji": korean_romaji, "kana": korean_kana}
    return reg


READINGS = _build_registry()


# --- Script detection -------------------------------------------------------------
_LATIN = r"A-Za-z\u00c0-\u024f"
_SCRIPT_CHARS = {
    "ko": r"\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f",          # Hangul
    "ja": r"\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff",          # Kana + Kanji
    "zh": r"\u4e00-\u9fff",
    "hi": r"\u0900-\u097f", "mr": r"\u0900-\u097f", "ne": r"\u0900-\u097f", "sa": r"\u0900-\u097f",
    "bn": r"\u0980-\u09ff", "pa": r"\u0a00-\u0a7f", "gu": r"\u0a80-\u0aff", "ta": r"\u0b80-\u0bff",
    "te": r"\u0c00-\u0c7f", "kn": r"\u0c80-\u0cff", "ml": r"\u0d00-\u0d7f",
}
for _l in ("en", "fr", "de", "es", "it", "pt", "nl", "sv", "da", "fi", "pl", "cs", "id", "vi", "tr"):
    _SCRIPT_CHARS[_l] = _LATIN
_SCRIPT_RE = {lang: re.compile(f"[{chars}]") for lang, chars in _SCRIPT_CHARS.items()}


def has_script(text, lang):
    """True if `text` contains characters of `lang`'s script (True for unknown languages)."""
    pattern = _SCRIPT_RE.get(lang)
    return True if pattern is None else bool(pattern.search(text))


def relabel_by_script(text, lang, candidates):
    """
    Whisper sometimes writes the *other* language even when forced (e.g. Japanese text
    under language="ko"). If the text has none of `lang`'s script but does contain the
    script of another candidate language, return that language; otherwise `lang`.
    """
    if lang not in _SCRIPT_RE or has_script(text, lang):
        return lang
    for other in candidates:
        if other != lang and other in _SCRIPT_RE and has_script(text, other):
            return other
    return lang


def make_reading(text, lang, style):
    """
    Returns the reading string for `text` (without parentheses), or None.
    style: "romaji" | "kana" | "both" | "none".
    If the requested style is unavailable for the language, falls back to romaji.
    No reading is produced unless the text contains the language's script (e.g. digits only).
    """
    funcs = READINGS.get(lang)
    if not funcs or style == "none" or not has_script(text, lang):
        return None
    wanted = ["romaji", "kana"] if style == "both" else [style]
    parts = [funcs[s](text) for s in wanted if s in funcs]
    if not parts and "romaji" in funcs:
        parts = [funcs["romaji"](text)]
    return " / ".join(parts) if parts else None
