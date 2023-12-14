import re
import pykakasi
from num2words import num2words

hira_mapper = [
    ("ゔ","bu"),
    # Conversion of 2 letters
    ("あぁ","a:"),
    ("いぃ","i:"),
    ("いぇ","je"),
    ("いゃ","ja"),
    ("うぅ","u:"),
    ("えぇ","e:"),
    ("おぉ","o:"),
    ("かぁ","ka:"),
    ("きぃ","ki:"),
    ("くぅ","ku:"),
    ("くゃ","kya"),
    ("くゅ","kyu"),
    ("くょ","kyo"),
    ("けぇ","ke:"),
    ("こぉ","ko:"),
    ("がぁ","ga:"),
    ("ぎぃ","gi:"),
    ("ぐぅ","gu:"),
    ("ぐゃ","gya"),
    ("ぐゅ","gyu"),
    ("ぐょ","gyo"),
    ("げぇ","ge:"),
    ("ごぉ","go:"),
    ("さぁ","sa:"),
    ("しぃ","shi:"),
    ("すぅ","su:"),
    ("すゃ","sha"),
    ("すゅ","shu"),
    ("すょ","sho"),
    ("せぇ","se:"),
    ("そぉ","so:"),
    ("ざぁ","za:"),
    ("じぃ","ji:"),
    ("ずぅ","zu:"),
    ("ずゃ","zya"),
    ("ずゅ","zyu"),
    ("ずょ","zyo"),
    ("ぜぇ","ze:"),
    ("ぞぉ","zo:"),
    ("たぁ","ta:"),
    ("ちぃ","chi:"),
    ("つぁ","tsa"),
    ("つぃ","tsi"),
    ("つぅ","tsu:"),
    ("つゃ","cha"),
    ("つゅ","chu"),
    ("つょ","cho"),
    ("つぇ","tse"),
    ("つぉ","tso"),
    ("てぇ","te:"),
    ("とぉ","to:"),
    ("だぁ","da:"),
    ("ぢぃ","ji:"),
    ("づぅ","du:"),
    ("づゃ","zya"),
    ("づゅ","zyu"),
    ("づょ","zyo"),
    ("でぇ","de:"),
    ("どぉ","do:"),
    ("なぁ","na:"),
    ("にぃ","ni:"),
    ("ぬぅ","nu:"),
    ("ぬゃ","nya"),
    ("ぬゅ","nyu"),
    ("ぬょ","nyo"),
    ("ねぇ","ne:"),
    ("のぉ","no:"),
    ("はぁ","ha:"),
    ("ひぃ","hi:"),
    ("ふぅ","fu:"),
    ("ふゃ","hya"),
    ("ふゅ","hyu"),
    ("ふょ","hyo"),
    ("へぇ","he:"),
    ("ほぉ","ho:"),
    ("ばぁ","ba:"),
    ("びぃ","bi:"),
    ("ぶぅ","bu:"),
    ("ぶゅ","byu"),
    ("べぇ","be:"),
    ("ぼぉ","bo:"),
    ("ぱぁ","pa:"),
    ("ぴぃ","pi:"),
    ("ぷぅ","pu:"),
    ("ぷゃ","pya"),
    ("ぷゅ","pyu"),
    ("ぷょ","pyo"),
    ("ぺぇ","pe:"),
    ("ぽぉ","po:"),
    ("まぁ","ma:"),
    ("みぃ","mi:"),
    ("むぅ","mu:"),
    ("むゃ","mya"),
    ("むゅ","myu"),
    ("むょ","myo"),
    ("めぇ","me:"),
    ("もぉ","mo:"),
    ("やぁ","ya:"),
    ("ゆぅ","yu:"),
    ("ゆゃ","ya:"),
    ("ゆゅ","yu:"),
    ("ゆょ","yo:"),
    ("よぉ","yo:"),
    ("らぁ","ra:"),
    ("りぃ","ri:"),
    ("るぅ","ru:"),
    ("るゃ","rya"),
    ("るゅ","ryu"),
    ("るょ","ryo"),
    ("れぇ","re:"),
    ("ろぉ","ro:"),
    ("わぁ","ua:"),
    ("をぉ","o:"),
    ("う゛","bu"),
    ("でぃ","di"),
    ("でゃ","dya"),
    ("でゅ","dyu"),
    ("でょ","dyo"),
    ("てぃ","ti"),
    ("てゃ","tya"),
    ("てゅ","tyu"),
    ("てょ","tyo"),
    ("すぃ","si"),
    ("ずぁ","zua"),
    ("ずぃ","zi"),
    ("ずぇ","ze"),
    ("ずぉ","zo"),
    ("きゃ","kya"),
    ("きゅ","kyu"),
    ("きょ","kyo"),
    ("しゃ","sha"),
    ("しゅ","shu"),
    ("しぇ","she"),
    ("しょ","sho"),
    ("ちゃ","cha"),
    ("ちゅ","chu"),
    ("ちぇ","che"),
    ("ちょ","cho"),
    ("とぅ","tu"),
    ("とゃ","tya"),
    ("とゅ","tyu"),
    ("とょ","tyo"),
    ("どぁ","doa"),
    ("どぅ","du"),
    ("どゃ","dya"),
    ("どゅ","dyu"),
    ("どょ","dyo"),
    ("にゃ","nya"),
    ("にゅ","nyu"),
    ("にょ","nyo"),
    ("ひゃ","hya"),
    ("ひゅ","hyu"),
    ("ひょ","hyo"),
    ("みゃ","mya"),
    ("みゅ","myu"),
    ("みょ","myo"),
    ("りゃ","rya"),
    ("りぇ","rye"),
    ("りゅ","ryu"),
    ("りょ","ryo"),
    ("ぎゃ","gya"),
    ("ぎゅ","gyu"),
    ("ぎょ","gyo"),
    ("ぢぇ","je"),
    ("ぢゃ","ja"),
    ("ぢゅ","ju"),
    ("ぢょ","jo"),
    ("じぇ","je"),
    ("じゃ","ja"),
    ("じゅ","ju"),
    ("じょ","jo"),
    ("びゃ","bya"),
    ("びゅ","byu"),
    ("びょ","byo"),
    ("ぴゃ","pya"),
    ("ぴゅ","pyu"),
    ("ぴょ","pyo"),
    ("うぁ","ua"),
    ("うぃ","wi"),
    ("うぇ","we"),
    ("うぉ","wo"),
    ("うゃ","wya"),
    ("うゅ","wyu"),
    ("うょ","wyo"),
    ("ふぁ","fa"),
    ("ふぃ","fi"),
    ("ふぇ","fe"),
    ("ふぉ","fo"),
    ("ゔぁ","ba"),
    ("ゔぃ","bi"),
    ("ゔぇ","be"),
    ("ゔぉ","bo"),
    ("ゔゃ","bya"),
    ("ゔゅ","byu"),
    ("ゔょ","byo"),
    # Conversion of 1 letter
    ("あ","a"),
    ("い","i"),
    ("う","u"),
    ("え","e"),
    ("お","o"),
    ("か","ka"),
    ("き","ki"),
    ("く","ku"),
    ("け","ke"),
    ("こ","ko"),
    ("さ","sa"),
    ("し","shi"),
    ("す","su"),
    ("せ","se"),
    ("そ","so"),
    ("た","ta"),
    ("ち","chi"),
    ("つ","tsu"),
    ("て","te"),
    ("と","to"),
    ("な","na"),
    ("に","ni"),
    ("ぬ","nu"),
    ("ね","ne"),
    ("の","no"),
    ("は","ha"),
    ("ひ","hi"),
    ("ふ","fu"),
    ("へ","he"),
    ("ほ","ho"),
    ("ま","ma"),
    ("み","mi"),
    ("む","mu"),
    ("め","me"),
    ("も","mo"),
    ("ら","ra"),
    ("り","ri"),
    ("る","ru"),
    ("れ","re"),
    ("ろ","ro"),
    ("が","ga"),
    ("ぎ","gi"),
    ("ぐ","gu"),
    ("げ","ge"),
    ("ご","go"),
    ("ざ","za"),
    ("じ","ji"),
    ("ず","zu"),
    ("ぜ","ze"),
    ("ぞ","zo"),
    ("だ","da"),
    ("ぢ","ji"),
    ("づ","zu"),
    ("で","de"),
    ("ど","do"),
    ("ば","ba"),
    ("び","bi"),
    ("ぶ","bu"),
    ("べ","be"),
    ("ぼ","bo"),
    ("ぱ","pa"),
    ("ぴ","pi"),
    ("ぷ","pu"),
    ("ぺ","pe"),
    ("ぽ","po"),
    ("や","ya"),
    ("ゆ","yu"),
    ("よ","yo"),
    ("わ","wa"),
    ("ゐ","i"),
    ("ゑ","e"),
    ("を","o"),
    ("ん","N"),
    ("っ","q"),
    ("ー",":"),

    # fix broken phonemes
    ("ぁ","a"),
    ("ぃ","i"),
    ("ぅ","u"),
    ("ぇ","e"),
    ("ぉ","o"),
    ("ゎ","wa"),
    ('ゃ', "ya"),
    ('ゅ', "yu"),
    ('ょ', "yo"),
    ("ゖ","ke"),
    ("ゕ", "ka")
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
        elif phoneme == "u" and pre[-1] == "o":  # uo
            combines.append(pre + "u")
            pre = None
        else:
            combines.append(pre)
            pre = phoneme

    if pre is not None: combines.append(pre)
    return combines

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