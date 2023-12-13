import re
import pykakasi
from num2words import num2words

hira_mapper = [
    ("ゔぁ","bˈa"),
    ("ゔぃ","bˈi"),
    ("ゔぇ","bˈe"),
    ("ゔぉ","bˈo"),
    ("ゔゃ","bˈʲa"),
    ("ゔゅ","bˈʲɯ"),
    ("ゔゃ","bˈʲa"),
    ("ゔょ","bˈʲo"),

    ("ゔ","bˈɯ"),

    ("あぁ","aː"),
    ("いぃ","iː"),
    ("いぇ","je"),
    ("いゃ","ja"),
    ("うぅ","ɯː"),
    ("えぇ","eː"),
    ("おぉ","oː"),
    ("かぁ","kˈaː"),
    ("きぃ","kˈiː"),
    ("くぅ","kˈɯː"),
    ("くゃ","kˈa"),
    ("くゅ","kˈʲɯ"),
    ("くょ","kˈʲo"),
    ("けぇ","kˈeː"),
    ("こぉ","kˈoː"),
    ("がぁ","gˈaː"),
    ("ぎぃ","gˈiː"),
    ("ぐぅ","gˈɯː"),
    ("ぐゃ","gˈʲa"),
    ("ぐゅ","gˈʲɯ"),
    ("ぐょ","gˈʲo"),
    ("げぇ","gˈeː"),
    ("ごぉ","gˈoː"),
    ("さぁ","sˈaː"),
    ("しぃ","ɕˈiː"),
    ("すぅ","sˈɯː"),
    ("すゃ","sˈʲa"),
    ("すゅ","sˈʲɯ"),
    ("すょ","sˈʲo"),
    ("せぇ","sˈeː"),
    ("そぉ","sˈoː"),
    ("ざぁ","zˈaː"),
    ("じぃ","dʑˈiː"),
    ("ずぅ","zˈɯː"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ぜぇ","zˈeː"),
    ("ぞぉ","zˈeː"),
    ("たぁ","tˈaː"),
    ("ちぃ","tɕˈiː"),
    ("つぁ","tsˈa"),
    ("つぃ","tsˈi"),
    ("つぅ","tsˈɯː"),
    ("つゃ","tɕˈa"),
    ("つゅ","tɕˈɯ"),
    ("つょ","tɕˈo"),
    ("つぇ","tsˈe"),
    ("つぉ","tsˈo"),
    ("てぇ","tˈeː"),
    ("とぉ","tˈoː"),
    ("だぁ","dˈaː"),
    ("ぢぃ","dʑˈiː"),
    ("づぅ","dˈɯː"),
    ("づゃ","zˈʲa"),
    ("づゅ","zˈʲɯ"),
    ("づょ","zˈʲo"),
    ("でぇ","dˈeː"),
    ("どぉ","dˈoː"),
    ("なぁ","nˈaː"),
    ("にぃ","nˈiː"),
    ("ぬぅ","nˈɯː"),
    ("ぬゃ","nˈʲa"),
    ("ぬゅ","nˈʲɯ"),
    ("ぬょ","nˈʲo"),
    ("ねぇ","nˈeː"),
    ("のぉ","nˈoː"),
    ("はぁ","hˈaː"),
    ("ひぃ","çˈiː"),
    ("ふぅ","ɸˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("へぇ","hˈeː"),
    ("ほぉ","hˈoː"),
    ("ばぁ","bˈaː"),
    ("びぃ","bˈiː"),
    ("ぶぅ","bˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ぶゅ","bˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("べぇ","bˈeː"),
    ("ぼぉ","bˈoː"),
    ("ぱぁ","pˈaː"),
    ("ぴぃ","pˈiː"),
    ("ぷぅ","pˈɯː"),
    ("ぷゃ","pˈʲa"),
    ("ぷゅ","pˈʲɯ"),
    ("ぷょ","pˈʲo"),
    ("ぺぇ","pˈeː"),
    ("ぽぉ","pˈoː"),
    ("まぁ","mˈaː"),
    ("みぃ","mˈiː"),
    ("むぅ","mˈɯː"),
    ("むゃ","mˈʲa"),
    ("むゅ","mˈʲɯ"),
    ("むょ","mˈʲo"),
    ("めぇ","mˈeː"),
    ("もぉ","mˈoː"),
    ("やぁ","jˈaː"),
    ("ゆぅ","jˈɯː"),
    ("ゆゃ","jˈaː"),
    ("ゆゅ","jˈɯː"),
    ("ゆょ","jˈoː"),
    ("よぉ","jˈoː"),
    ("らぁ","ɽˈaː"),
    ("りぃ","ɽˈiː"),
    ("るぅ","ɽˈɯː"),
    ("るゃ","ɽˈʲa"),
    ("るゅ","ɽˈʲɯ"),
    ("るょ","ɽˈʲo"),
    ("れぇ","ɽˈeː"),
    ("ろぉ","ɽˈoː"),
    ("わぁ","ɯˈaː"),
    ("をぉ","oː"),

    ("う゛","bˈɯ"),
    ("でぃ","dˈi"),
    ("でぇ","dˈeː"),
    ("でゃ","dˈʲa"),
    ("でゅ","dˈʲɯ"),
    ("でょ","dˈʲo"),
    ("てぃ","tˈi"),
    ("てぇ","tˈeː"),
    ("てゃ","tˈʲa"),
    ("てゅ","tˈʲɯ"),
    ("てょ","tˈʲo"),
    ("すぃ","sˈi"),
    ("ずぁ","zˈɯa"),
    ("ずぃ","zˈi"),
    ("ずぅ","zˈɯ"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ずぇ","zˈe"),
    ("ずぉ","zˈo"),
    ("きゃ","kˈʲa"),
    ("きゅ","kˈʲɯ"),
    ("きょ","kˈʲo"),
    ("しゃ","ɕˈʲa"),
    ("しゅ","ɕˈʲɯ"),
    ("しぇ","ɕˈʲe"),
    ("しょ","ɕˈʲo"),
    ("ちゃ","tɕˈa"),
    ("ちゅ","tɕˈɯ"),
    ("ちぇ","tɕˈe"),
    ("ちょ","tɕˈo"),
    ("とぅ","tˈɯ"),
    ("とゃ","tˈʲa"),
    ("とゅ","tˈʲɯ"),
    ("とょ","tˈʲo"),
    ("どぁ","dˈoa"),
    ("どぅ","dˈɯ"),
    ("どゃ","dˈʲa"),
    ("どゅ","dˈʲɯ"),
    ("どょ","dˈʲo"),
    ("どぉ","dˈoː"),
    ("にゃ","nˈʲa"),
    ("にゅ","nˈʲɯ"),
    ("にょ","nˈʲo"),
    ("ひゃ","çˈʲa"),
    ("ひゅ","çˈʲɯ"),
    ("ひょ","çˈʲo"),
    ("みゃ","mˈʲa"),
    ("みゅ","mˈʲɯ"),
    ("みょ","mˈʲo"),
    ("りゃ","ɽˈʲa"),
    ("りぇ","ɽˈʲe"),
    ("りゅ","ɽˈʲɯ"),
    ("りょ","ɽˈʲo"),
    ("ぎゃ","gˈʲa"),
    ("ぎゅ","gˈʲɯ"),
    ("ぎょ","gˈʲo"),
    ("ぢぇ","dʑˈe"),
    ("ぢゃ","dʑˈa"),
    ("ぢゅ","dʑˈɯ"),
    ("ぢょ","dʑˈo"),
    ("じぇ","dʑˈe"),
    ("じゃ","dʑˈa"),
    ("じゅ","dʑˈɯ"),
    ("じょ","dʑˈo"),
    ("びゃ","bˈʲa"),
    ("びゅ","bˈʲɯ"),
    ("びょ","bˈʲo"),
    ("ぴゃ","pˈʲa"),
    ("ぴゅ","pˈʲɯ"),
    ("ぴょ","pˈʲo"),
    ("うぁ","ɯˈa"),
    ("うぃ","ɯˈi"),
    ("うぇ","ɯˈe"),
    ("うぉ","ɯˈo"),
    ("うゃ","ɯˈʲa"),
    ("うゅ","ɯˈʲɯ"),
    ("うょ","ɯˈʲo"),
    ("ふぁ","ɸˈa"),
    ("ふぃ","ɸˈi"),
    ("ふぅ","ɸˈɯ"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("ふぇ","ɸˈe"),
    ("ふぉ","ɸˈo"),

    ("あ","a"),
    ("い","i"),
    ("う","ɯ"),
    ("え","e"),
    ("お","o"),
    ("か","kˈa"),
    ("き","kˈi"),
    ("く","kˈɯ"),
    ("け","kˈe"),
    ("こ","kˈo"),
    ("さ","sˈa"),
    ("し","ɕˈi"),
    ("す","sˈɯ"),
    ("せ","sˈe"),
    ("そ","sˈo"),
    ("た","tˈa"),
    ("ち","tɕˈi"),
    ("つ","tsˈɯ"),
    ("て","tˈe"),
    ("と","tˈo"),
    ("な","nˈa"),
    ("に","nˈi"),
    ("ぬ","nˈɯ"),
    ("ね","nˈe"),
    ("の","nˈo"),
    ("は","hˈa"),
    ("ひ","çˈi"),
    ("ふ","ɸˈɯ"),
    ("へ","hˈe"),
    ("ほ","hˈo"),
    ("ま","mˈa"),
    ("み","mˈi"),
    ("む","mˈɯ"),
    ("め","mˈe"),
    ("も","mˈo"),
    ("ら","ɽˈa"),
    ("り","ɽˈi"),
    ("る","ɽˈɯ"),
    ("れ","ɽˈe"),
    ("ろ","ɽˈo"),
    ("が","gˈa"),
    ("ぎ","gˈi"),
    ("ぐ","gˈɯ"),
    ("げ","gˈe"),
    ("ご","gˈo"),
    ("ざ","zˈa"),
    ("じ","dʑˈi"),
    ("ず","zˈɯ"),
    ("ぜ","zˈe"),
    ("ぞ","zˈo"),
    ("だ","dˈa"),
    ("ぢ","dʑˈi"),
    ("づ","zˈɯ"),
    ("で","dˈe"),
    ("ど","dˈo"),
    ("ば","bˈa"),
    ("び","bˈi"),
    ("ぶ","bˈɯ"),
    ("べ","bˈe"),
    ("ぼ","bˈo"),
    ("ぱ","pˈa"),
    ("ぴ","pˈi"),
    ("ぷ","pˈɯ"),
    ("ぺ","pˈe"),
    ("ぽ","pˈo"),
    ("や","jˈa"),
    ("ゆ","jˈɯ"),
    ("よ","jˈo"),
    ("わ","ɯˈa"),
    ("ゐ","i"),
    ("ゑ","e"),
    ("ん","ɴ"),
    ("っ","ʔ"),
    ("ー","ː"),

    # fix broken phonemes
    ("ぁ","a"),
    ("ぃ","i"),
    ("ぅ","ɯ"),
    ("ぇ","e"),
    ("ぉ","o"),
    ("ゎ","ɯˈa"),
    ("ぉ","o"),
    ('ゃ', "ʲa"),
    ('ゅ', "ʲɯ"),
    ('ょ', "ʲo"),
    ("を","o"),
    ("ゖ","kˈe"),
    ("ゕ", "kˈa")
]

# 鼻音
nasal_sound = [
    # before m, p, b
    ("ɴm","mm"),
    ("ɴb", "mb"),
    ("ɴp", "mp"),

    # before k, g
    ("ɴk","ŋk"),
    ("ɴg", "ŋg"),

    # before t, d, n, s, z, ɽ
    ("ɴt","nt"),
    ("ɴd", "nd"),
    ("ɴn","nn"),
    ("ɴs", "ns"),
    ("ɴz","nz"),
    ("ɴɽ", "nɽ"),

    ("ɴɲ", "ɲɲ"),
]

_RULEMAP1 = {k: v for k, v in hira_mapper if len(k) == 1}
_RULEMAP2 = {k: v for k, v in hira_mapper if len(k) == 2}

def hira2ipa(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()
    phonemes = []
    while text:
        if len(text) >= 2:
            x = _RULEMAP2.get(text[:2])
            if x is not None:
                text = text[2:]
                phonemes.append(x)
                continue
        x = _RULEMAP1.get(text[0])
        if x is not None:
            text = text[1:]
            phonemes.append(x)
            continue

        phonemes.append(text[0])
        text = text[1:]
    return phonemes

# 将一些意外分开的音标合并
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
        elif phoneme == "i" and (pre[-1] == "a" or pre[-1] == "e"):  # ai, ei
            combines.append(pre + "i")
            pre = None
        elif phoneme == "ɯ" and pre[-1] == "o":  # uo
            combines.append(pre + "ɯ")
            pre = None
        else:
            combines.append(pre)
            pre = phoneme

    if pre is not None: combines.append(pre)
    return combines

def replace_nasal(ptext:str) -> str:
    for item in nasal_sound:
        k, v = item
        ptext = ptext.replace(k, v)
    return ptext

def separate_punctuation(ptext:str) -> list:
    pos = len(ptext)
    while pos > 0 and ptext[pos - 1] in [',', '.', '?', '!']:
        pos -= 1
    if pos == len(ptext):
        return [ptext]
    else:
        return [ptext[0:pos], ptext[pos:len(ptext)]]

def repair_hira_texts(texts:list) -> list:
    '''# sometime hira text separation will be wrong, sub-character like 'ゃ' will be the first char, fix it by combine
    hira text starts with sub-character like 'ゃ'
    '''
    if len(texts) == 0: return texts
    new_texts = [texts[0]]
    for text in texts[1:]:
        if text[0] in ['ぁ', 'ぃ', 'ぇ', 'ぉ', 'ゃ', 'ゅ', 'ょ', 'ぅ', '゛']:
            new_texts[-1] = new_texts[-1] + text
        else:
            new_texts.append(text)
    return new_texts

def text_to_phonemes(text:str, sep="", phase_sep=" ") -> str:
    kks = pykakasi.kakasi()
    converted = kks.convert(text)

    hira_texts = [item["hira"] for item in converted]
    hira_texts = repair_hira_texts(hira_texts)

    ptexts = []
    for hira in hira_texts:
        phonemes = hira2ipa(hira)
        phonemes = combine_phonemes(phonemes)
        ptext = sep.join(phonemes)
        ptext = re.sub(r":+", ":", ptext)
        ptext = replace_nasal(ptext)
        ptext_list = separate_punctuation(ptext)
        ptexts.extend(ptext_list)

    ptexts = [p for p in ptexts if len(p.strip()) > 0]
    phonemes_text = phase_sep.join(ptexts)
    phonemes_text = re.sub(r" +", " ", phonemes_text)
    return phonemes_text

_ALPHASYMBOL_YOMI = {
    '#': 'しゃーぷ',
    '%': 'ぱーせんと',
    '&': 'あんど',
    '+': 'ぷらす',
    '-': 'まいなす',
    ':': 'ころん',
    ';': 'せみころん',
    '<': 'しょう',
    '=': 'いこーる',
    '>': 'だい',
    '@': 'あっと',
    'a': 'えー',
    'b': 'びー',
    'c': 'しー',
    'd': 'でぃー',
    'e': 'いー',
    'f': 'えふ',
    'g': 'じー',
    'h': 'えいち',
    'i': 'あい',
    'j': 'じぇー',
    'k': 'けー',
    'l': 'える',
    'm': 'えむ',
    'n': 'えぬ',
    'o': 'おー',
    'p': 'ぴー',
    'q': 'きゅー',
    'r': 'あーる',
    's': 'えす',
    't': 'てぃー',
    'u': 'ゆー',
    'v': 'ぶい',
    'w': 'だぶりゅー',
    'x': 'えっくす',
    'y': 'わい',
    'z': 'ぜっと',
    'α': 'あるふ',
    'β': 'べーた',
    'γ': 'がんま',
    'δ': 'でるた',
    'ε': 'いぷしろん',
    'ζ': 'ぜーた',
    'η': 'いーた',
    'θ': 'しーた',
    'ι': 'いおた',
    'κ': 'かっぱ',
    'λ': 'らむだ',
    'μ': 'みゅー',
    'ν': 'にゅー',
    'ξ': 'くさい',
    'ο': 'おみくろん',
    'π': 'ぱい',
    'ρ': 'ろー',
    'σ': 'しぐま',
    'τ': 'たう',
    'υ': 'うぷしろん',
    'φ': 'ふ',
    'χ': 'かい',
    'ψ': 'ぷさい',
    'ω': 'おめが'
}

_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "どる", "¥": "えん", "£": "ぽんど", "€": "ゆーろ"}
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
        if char in ['々', 'ヽ', 'ヾ', 'ゝ', 'ゞ', '〃']:
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
        # sep=" ", phase_sep=" sp "
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