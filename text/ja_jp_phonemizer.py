import re
import MeCab
import pykakasi
from num2words import num2words

_CONVRULES = [
    # Conversion of 2 letters
    "アァ/ a a",
    "イィ/ i i",
    "イェ/ i e",
    "イャ/ y a",
    "ウゥ/ u:",
    "エェ/ e e",
    "オォ/ o:",
    "カァ/ k a:",
    "キィ/ k i:",
    "クゥ/ k u:",
    "クャ/ ky a",
    "クュ/ ky u",
    "クョ/ ky o",
    "ケェ/ k e:",
    "コォ/ k o:",
    "ガァ/ g a:",
    "ギィ/ g i:",
    "グゥ/ g u:",
    "グャ/ gy a",
    "グュ/ gy u",
    "グョ/ gy o",
    "ゲェ/ g e:",
    "ゴォ/ g o:",
    "サァ/ s a:",
    "シィ/ sh i:",
    "スゥ/ s u:",
    "スャ/ sh a",
    "スュ/ sh u",
    "スョ/ sh o",
    "セェ/ s e:",
    "ソォ/ s o:",
    "ザァ/ z a:",
    "ジィ/ j i:",
    "ズゥ/ z u:",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ゼェ/ z e:",
    "ゾォ/ z o:",
    "タァ/ t a:",
    "チィ/ ch i:",
    "ツァ/ ts a",
    "ツィ/ ts i",
    "ツゥ/ ts u:",
    "ツャ/ ch a",
    "ツュ/ ch u",
    "ツョ/ ch o",
    "ツェ/ ts e",
    "ツォ/ ts o",
    "テェ/ t e:",
    "トォ/ t o:",
    "ダァ/ d a:",
    "ヂィ/ j i:",
    "ヅゥ/ d u:",
    "ヅャ/ zy a",
    "ヅュ/ zy u",
    "ヅョ/ zy o",
    "デェ/ d e:",
    "ドォ/ d o:",
    "ナァ/ n a:",
    "ニィ/ n i:",
    "ヌゥ/ n u:",
    "ヌャ/ ny a",
    "ヌュ/ ny u",
    "ヌョ/ ny o",
    "ネェ/ n e:",
    "ノォ/ n o:",
    "ハァ/ h a:",
    "ヒィ/ h i:",
    "フゥ/ f u:",
    "フャ/ hy a",
    "フュ/ hy u",
    "フョ/ hy o",
    "ヘェ/ h e:",
    "ホォ/ h o:",
    "バァ/ b a:",
    "ビィ/ b i:",
    "ブゥ/ b u:",
    "フャ/ hy a",
    "ブュ/ by u",
    "フョ/ hy o",
    "ベェ/ b e:",
    "ボォ/ b o:",
    "パァ/ p a:",
    "ピィ/ p i:",
    "プゥ/ p u:",
    "プャ/ py a",
    "プュ/ py u",
    "プョ/ py o",
    "ペェ/ p e:",
    "ポォ/ p o:",
    "マァ/ m a:",
    "ミィ/ m i:",
    "ムゥ/ m u:",
    "ムャ/ my a",
    "ムュ/ my u",
    "ムョ/ my o",
    "メェ/ m e:",
    "モォ/ m o:",
    "ヤァ/ y a:",
    "ユゥ/ y u:",
    "ユャ/ y a:",
    "ユュ/ y u:",
    "ユョ/ y o:",
    "ヨォ/ y o:",
    "ラァ/ r a:",
    "リィ/ r i:",
    "ルゥ/ r u:",
    "ルャ/ ry a",
    "ルュ/ ry u",
    "ルョ/ ry o",
    "レェ/ r e:",
    "ロォ/ r o:",
    "ワァ/ w a:",
    "ヲォ/ o:",
    "ディ/ d i",
    "デェ/ d e:",
    "デャ/ dy a",
    "デュ/ dy u",
    "デョ/ dy o",
    "ティ/ t i",
    "テェ/ t e:",
    "テャ/ ty a",
    "テュ/ ty u",
    "テョ/ ty o",
    "スィ/ s i",
    "ズァ/ z u a",
    "ズィ/ z i",
    "ズゥ/ z u",
    "ズャ/ zy a",
    "ズュ/ zy u",
    "ズョ/ zy o",
    "ズェ/ z e",
    "ズォ/ z o",
    "キャ/ ky a",
    "キュ/ ky u",
    "キョ/ ky o",
    "シャ/ sh a",
    "シュ/ sh u",
    "シェ/ sh e",
    "ショ/ sh o",
    "チャ/ ch a",
    "チュ/ ch u",
    "チェ/ ch e",
    "チョ/ ch o",
    "トゥ/ t u",
    "トャ/ ty a",
    "トュ/ ty u",
    "トョ/ ty o",
    "ドァ/ d o a",
    "ドゥ/ d u",
    "ドャ/ dy a",
    "ドュ/ dy u",
    "ドョ/ dy o",
    "ドォ/ d o:",
    "ニャ/ ny a",
    "ニュ/ ny u",
    "ニョ/ ny o",
    "ヒャ/ hy a",
    "ヒュ/ hy u",
    "ヒョ/ hy o",
    "ミャ/ my a",
    "ミュ/ my u",
    "ミョ/ my o",
    "リャ/ ry a",
    "リュ/ ry u",
    "リョ/ ry o",
    "ギャ/ gy a",
    "ギュ/ gy u",
    "ギョ/ gy o",
    "ヂェ/ j e",
    "ヂャ/ j a",
    "ヂュ/ j u",
    "ヂョ/ j o",
    "ジェ/ j e",
    "ジャ/ j a",
    "ジュ/ j u",
    "ジョ/ j o",
    "ビャ/ by a",
    "ビュ/ by u",
    "ビョ/ by o",
    "ピャ/ py a",
    "ピュ/ py u",
    "ピョ/ py o",
    "ウァ/ u a",
    "ウィ/ w i",
    "ウェ/ w e",
    "ウォ/ w o",
    "ファ/ f a",
    "フィ/ f i",
    "フゥ/ f u",
    "フャ/ hy a",
    "フュ/ hy u",
    "フョ/ hy o",
    "フェ/ f e",
    "フォ/ f o",
    "ヴァ/ b a",
    "ヴィ/ b i",
    "ヴェ/ b e",
    "ヴォ/ b o",
    "ヴュ/ by u",
    # Conversion of 1 letter
    "ア/ a",
    "イ/ i",
    "ウ/ u",
    "エ/ e",
    "オ/ o",
    "カ/ k a",
    "キ/ k i",
    "ク/ k u",
    "ケ/ k e",
    "コ/ k o",
    "サ/ s a",
    "シ/ sh i",
    "ス/ s u",
    "セ/ s e",
    "ソ/ s o",
    "タ/ t a",
    "チ/ ch i",
    "ツ/ ts u",
    "テ/ t e",
    "ト/ t o",
    "ナ/ n a",
    "ニ/ n i",
    "ヌ/ n u",
    "ネ/ n e",
    "ノ/ n o",
    "ハ/ h a",
    "ヒ/ h i",
    "フ/ f u",
    "ヘ/ h e",
    "ホ/ h o",
    "マ/ m a",
    "ミ/ m i",
    "ム/ m u",
    "メ/ m e",
    "モ/ m o",
    "ラ/ r a",
    "リ/ r i",
    "ル/ r u",
    "レ/ r e",
    "ロ/ r o",
    "ガ/ g a",
    "ギ/ g i",
    "グ/ g u",
    "ゲ/ g e",
    "ゴ/ g o",
    "ザ/ z a",
    "ジ/ j i",
    "ズ/ z u",
    "ゼ/ z e",
    "ゾ/ z o",
    "ダ/ d a",
    "ヂ/ j i",
    "ヅ/ z u",
    "デ/ d e",
    "ド/ d o",
    "バ/ b a",
    "ビ/ b i",
    "ブ/ b u",
    "ベ/ b e",
    "ボ/ b o",
    "パ/ p a",
    "ピ/ p i",
    "プ/ p u",
    "ペ/ p e",
    "ポ/ p o",
    "ヤ/ y a",
    "ユ/ y u",
    "ヨ/ y o",
    "ワ/ w a",
    "ヰ/ i",
    "ヱ/ e",
    "ヲ/ o",
    "ン/ N",
    "ッ/ q",
    "ヴ/ b u",
    "ー/:",
    # Try converting broken text
    "ァ/ a",
    "ィ/ i",
    "ゥ/ u",
    "ェ/ e",
    "ォ/ o",
    "ヮ/ w a",
    "ォ/ o",
    "ヵ/ k a",
    "ャ/ y a",
    "ュ/ y u",
    "ョ/ y o",
    "ヶ/ g a",
    # Symbols
    # "、/ ,",
    # "。/ .",
    # "！/ !",
    # "？/ ?",
    # "・/ ,",
]

def _makerulemap():
    l = [tuple(x.split("/")) for x in _CONVRULES]
    return tuple({k: v for k, v in l if len(k) == i} for i in (1, 2))
_RULEMAP1, _RULEMAP2 = _makerulemap()


def kana2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()
    phonemes = []
    while text:
        if len(text) >= 2:
            x = _RULEMAP2.get(text[:2])
            if x is not None:
                text = text[2:]
                phonemes.append(x.replace(' ', ''))
                continue
        x = _RULEMAP1.get(text[0])
        if x is not None:
            text = text[1:]
            phonemes.append(x.replace(' ', ''))
            continue

        phonemes.append(" ")
        phonemes.append(text[0])
        text = text[1:]
    return phonemes

def combine_phonemes(phonemes:list) -> list:
    pre = None
    combines = []
    for phoneme in phonemes:
        if pre is None:
            pre = phoneme
        elif phoneme == "N":
            combines.append(pre + "N")
            pre = None
        elif phoneme == ":":
            combines.append(pre + ":")
            pre = None
        elif phoneme == "i" and (pre[-1] == "a" or pre[-1] == "e"):
            combines.append(pre + "i")
            pre = None
        elif phoneme == "u" and pre[-1] == "o":
            combines.append(pre + "u")
            pre = None
        else:
            combines.append(pre)
            pre = phoneme

    if pre is not None: combines.append(pre)
    return combines

def text_to_phonemes(text:str, sep="", phase_sep=" ") -> str:
    kks = pykakasi.kakasi()
    converted = kks.convert(text)

    texts = [item["kana"] for item in converted]
    # print(" ".join(texts))

    ptexts = []
    for kana in texts:
        phonemes = kana2phoneme(kana)
        phonemes = combine_phonemes(phonemes)
        ptext = sep.join(phonemes)
        ptext = re.sub(r":+", ":", ptext)
        ptexts.append(ptext)

    ptexts = [p for p in ptexts if len(p.strip()) > 0]
    phonemes_text = phase_sep.join(ptexts)
    phonemes_text = re.sub(r" +", " ", phonemes_text)
    return phonemes_text

_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",
    ">": "大なり",
    "@": "アット",
    "a": "エー",
    "b": "ビー",
    "c": "シー",
    "d": "ディー",
    "e": "イー",
    "f": "エフ",
    "g": "ジー",
    "h": "エイチ",
    "i": "アイ",
    "j": "ジェー",
    "k": "ケー",
    "l": "エル",
    "m": "エム",
    "n": "エヌ",
    "o": "オー",
    "p": "ピー",
    "q": "キュー",
    "r": "アール",
    "s": "エス",
    "t": "ティー",
    "u": "ユー",
    "v": "ブイ",
    "w": "ダブリュー",
    "x": "エックス",
    "y": "ワイ",
    "z": "ゼット",
    "α": "アルファ",
    "β": "ベータ",
    "γ": "ガンマ",
    "δ": "デルタ",
    "ε": "イプシロン",
    "ζ": "ゼータ",
    "η": "イータ",
    "θ": "シータ",
    "ι": "イオタ",
    "κ": "カッパ",
    "λ": "ラムダ",
    "μ": "ミュー",
    "ν": "ニュー",
    "ξ": "クサイ",
    "ο": "オミクロン",
    "π": "パイ",
    "ρ": "ロー",
    "σ": "シグマ",
    "τ": "タウ",
    "υ": "ウプシロン",
    "φ": "ファイ",
    "χ": "カイ",
    "ψ": "プサイ",
    "ω": "オメガ",
}


_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")

def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])

def japanese_convert_double_to_words(text:str) -> str:
    out = ""
    for i, char in enumerate(text):
        if char == '々' or char == 'ゝ':
            out += text[i-1]  # use the char in left to replace 々
        else:
            out += char
    return out

def japanese_text_to_phonemes(text:str, sep="", phase_sep=" ") -> str:
    text = japanese_convert_numbers_to_words(text)
    text = japanese_convert_alpha_symbols_to_words(text)
    text = japanese_convert_double_to_words(text)
    text = text_to_phonemes(text, sep, phase_sep)
    return text

if __name__ == "__main__":
    text = japanese_text_to_phonemes(
        "中国人の健康状態は良好",
        # sep=" ", phase_sep=" sp "
    )
    print(text)
    text = japanese_text_to_phonemes(
        "でも実際は静かです",
        # sep=" ", phase_sep=" sp "
    )
    print(text)
    text = japanese_text_to_phonemes(
        "単語の発音は単語の意味に影響を与えません",
        sep=" ", phase_sep=" sp "
    )
    print(text)
    text = japanese_text_to_phonemes(
        "ケイキョウ",
        # sep=" ", phase_sep=" sp "
    )
    print(text)
    text = japanese_text_to_phonemes(
        "整理するのが難しく、ほとんど使用されないため、これ以上詳しくは説明しません",
        # sep=" ", phase_sep=" sp "
    )
    print(text)