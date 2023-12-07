from typing import List
import jieba
import pypinyin

PINYIN_DICT = {
    "a": ["a"],
    "ai": ["ai"],
    "an": ["an"],
    "ang": ["ɑŋ"],
    "ao": ["aʌ"],
    "ba": ["ba"],
    "bai": ["bai"],
    "ban": ["ban"],
    "bang": ["bɑŋ"],
    "bao": ["baʌ"],
    # "be": ["be"], doesnt exist
    "bei": ["bɛi"],
    "ben": ["bœn"],
    "beng": ["bɵŋ"],
    "bi": ["bi"],
    "bian": ["biɛn"],
    "biao": ["biaʌ"],
    "bie": ["bie"],
    "bin": ["bin"],
    "bing": ["bɨŋ"],
    "bo": ["bo"],
    "bu": ["bu"],
    "ca": ["tsa"],
    "cai": ["tsai"],
    "can": ["tsan"],
    "cang": ["tsɑŋ"],
    "cao": ["tsaʌ"],
    "ce": ["tsø"],
    "cen": ["tsœn"],
    "ceng": ["tsɵŋ"],
    "cha": ["ʈʂa"],
    "chai": ["ʈʂai"],
    "chan": ["ʈʂan"],
    "chang": ["ʈʂɑŋ"],
    "chao": ["ʈʂaʌ"],
    "che": ["ʈʂø"],
    "chen": ["ʈʂœn"],
    "cheng": ["ʈʂɵŋ"],
    "chi": ["ʈʂʏ"],
    "chong": ["ʈʂoŋ"],
    "chou": ["ʈʂou"],
    "chu": ["ʈʂu"],
    "chua": ["ʈʂua"],
    "chuai": ["ʈʂuai"],
    "chuan": ["ʈʂuan"],
    "chuang": ["ʈʂuɑŋ"],
    "chui": ["ʈʂuei"],
    "chun": ["ʈʂun"],
    "chuo": ["ʈʂuo"],
    "ci": ["tsɪ"],
    "cong": ["tsoŋ"],
    "cou": ["tsou"],
    "cu": ["tsu"],
    "cuan": ["tsuan"],
    "cui": ["tsuei"],
    "cun": ["tsun"],
    "cuo": ["tsuo"],
    "da": ["da"],
    "dai": ["dai"],
    "dan": ["dan"],
    "dang": ["dɑŋ"],
    "dao": ["daʌ"],
    "de": ["dø"],
    "dei": ["dei"],
    # "den": ["dœn"],
    "deng": ["dɵŋ"],
    "di": ["di"],
    "dia": ["dia"],
    "dian": ["diɛn"],
    "diao": ["diaʌ"],
    "die": ["die"],
    "ding": ["dɨŋ"],
    "diu": ["dio"],
    "dong": ["doŋ"],
    "dou": ["dou"],
    "du": ["du"],
    "duan": ["duan"],
    "dui": ["duei"],
    "dun": ["dun"],
    "duo": ["duo"],
    "e": ["ø"],
    "ei": ["ei"],
    "en": ["œn"],
    # "ng": ["œn"],
    # "eng": ["ɵŋ"],
    "er": ["er"],
    "fa": ["fa"],
    "fan": ["fan"],
    "fang": ["fɑŋ"],
    "fei": ["fei"],
    "fen": ["fœn"],
    "feng": ["fɵŋ"],
    "fo": ["fo"],
    "fou": ["fou"],
    "fu": ["fu"],
    "ga": ["ga"],
    "gai": ["gai"],
    "gan": ["gan"],
    "gang": ["gɑŋ"],
    "gao": ["gaʌ"],
    "ge": ["gø"],
    "gei": ["gei"],
    "gen": ["gœn"],
    "geng": ["gɵŋ"],
    "gong": ["goŋ"],
    "gou": ["gou"],
    "gu": ["gu"],
    "gua": ["gua"],
    "guai": ["guai"],
    "guan": ["guan"],
    "guang": ["guɑŋ"],
    "gui": ["guei"],
    "gun": ["gun"],
    "guo": ["guo"],
    "ha": ["xa"],
    "hai": ["xai"],
    "han": ["xan"],
    "hang": ["xɑŋ"],
    "hao": ["xaʌ"],
    "he": ["xø"],
    "hei": ["xei"],
    "hen": ["xœn"],
    "heng": ["xɵŋ"],
    "hong": ["xoŋ"],
    "hou": ["xou"],
    "hu": ["xu"],
    "hua": ["xua"],
    "huai": ["xuai"],
    "huan": ["xuan"],
    "huang": ["xuɑŋ"],
    "hui": ["xuei"],
    "hun": ["xun"],
    "huo": ["xuo"],
    "ji": ["dʑi"],
    "jia": ["dʑia"],
    "jian": ["dʑiɛn"],
    "jiang": ["dʑiɑŋ"],
    "jiao": ["dʑiaʌ"],
    "jie": ["dʑie"],
    "jin": ["dʑin"],
    "jing": ["dʑɨŋ"],
    "jiong": ["dʑioŋ"],
    "jiu": ["dʑio"],
    "ju": ["dʑy"],
    "juan": ["dʑyɛn"],
    "jue": ["dʑye"],
    "jun": ["dʑyn"],
    "ka": ["ka"],
    "kai": ["kai"],
    "kan": ["kan"],
    "kang": ["kɑŋ"],
    "kao": ["kaʌ"],
    "ke": ["kø"],
    "kei": ["kei"],
    "ken": ["kœn"],
    "keng": ["kɵŋ"],
    "kong": ["koŋ"],
    "kou": ["kou"],
    "ku": ["ku"],
    "kua": ["kua"],
    "kuai": ["kuai"],
    "kuan": ["kuan"],
    "kuang": ["kuɑŋ"],
    "kui": ["kuei"],
    "kun": ["kun"],
    "kuo": ["kuo"],
    "la": ["la"],
    "lai": ["lai"],
    "lan": ["lan"],
    "lang": ["lɑŋ"],
    "lao": ["laʌ"],
    "le": ["lø"],
    "lei": ["lei"],
    "leng": ["lɵŋ"],
    "li": ["li"],
    "lia": ["lia"],
    "lian": ["liɛn"],
    "liang": ["liɑŋ"],
    "liao": ["liaʌ"],
    "lie": ["lie"],
    "lin": ["lin"],
    "ling": ["lɨŋ"],
    "liu": ["lio"],
    "lo": ["lo"],
    "long": ["loŋ"],
    "lou": ["lou"],
    "lu": ["lu"],
    "lv": ["ly"],
    "luan": ["luan"],
    "lve": ["lye"],
    "lue": ["lue"],
    "lun": ["lun"],
    "luo": ["luo"],
    "ma": ["ma"],
    "mai": ["mai"],
    "man": ["man"],
    "mang": ["mɑŋ"],
    "mao": ["maʌ"],
    "me": ["mø"],
    "mei": ["mei"],
    "men": ["mœn"],
    "meng": ["mɵŋ"],
    "mi": ["mi"],
    "mian": ["miɛn"],
    "miao": ["miaʌ"],
    "mie": ["mie"],
    "min": ["min"],
    "ming": ["mɨŋ"],
    "miu": ["mio"],
    "mo": ["mo"],
    "mou": ["mou"],
    "mu": ["mu"],
    "na": ["na"],
    "nai": ["nai"],
    "nan": ["nan"],
    "nang": ["nɑŋ"],
    "nao": ["naʌ"],
    "ne": ["nø"],
    "nei": ["nei"],
    "nen": ["nœn"],
    "neng": ["nɵŋ"],
    "ni": ["ni"],
    "nia": ["nia"],
    "nian": ["niɛn"],
    "niang": ["niɑŋ"],
    "niao": ["niaʌ"],
    "nie": ["nie"],
    "nin": ["nin"],
    "ning": ["nɨŋ"],
    "niu": ["nio"],
    "nong": ["noŋ"],
    "nou": ["nou"],
    "nu": ["nu"],
    "nv": ["ny"],
    "nuan": ["nuan"],
    "nve": ["nye"],
    "nue": ["nye"],
    "nuo": ["nuo"],
    "o": ["o"],
    "ou": ["ou"],
    "pa": ["pa"],
    "pai": ["pai"],
    "pan": ["pan"],
    "pang": ["pɑŋ"],
    "pao": ["paʌ"],
    "pe": ["pø"],
    "pei": ["pei"],
    "pen": ["pœn"],
    "peng": ["pɵŋ"],
    "pi": ["pi"],
    "pian": ["piɛn"],
    "piao": ["piaʌ"],
    "pie": ["pie"],
    "pin": ["pin"],
    "ping": ["pɨŋ"],
    "po": ["po"],
    "pou": ["pou"],
    "pu": ["pu"],
    "qi": ["tɕi"],
    "qia": ["tɕia"],
    "qian": ["tɕiɛn"],
    "qiang": ["tɕiɑŋ"],
    "qiao": ["tɕiaʌ"],
    "qie": ["tɕie"],
    "qin": ["tɕin"],
    "qing": ["tɕɨŋ"],
    "qiong": ["tɕioŋ"],
    "qiu": ["tɕio"],
    "qu": ["tɕy"],
    "quan": ["tɕyɛn"],
    "que": ["tɕye"],
    "qun": ["tɕyn"],
    "ran": ["ʐan"],
    "rang": ["ʐɑŋ"],
    "rao": ["ʐaʌ"],
    "re": ["ʐø"],
    "ren": ["ʐœn"],
    "reng": ["ʐɵŋ"],
    "ri": ["ʐʏ"],
    "rong": ["ʐoŋ"],
    "rou": ["ʐou"],
    "ru": ["ʐu"],
    "rua": ["ʐua"],
    "ruan": ["ʐuan"],
    "rui": ["ʐuei"],
    "run": ["ʐun"],
    "ruo": ["ʐuo"],
    "sa": ["sa"],
    "sai": ["sai"],
    "san": ["san"],
    "sang": ["sɑŋ"],
    "sao": ["saʌ"],
    "se": ["sø"],
    "sen": ["sœn"],
    "seng": ["sɵŋ"],
    "sha": ["ʂa"],
    "shai": ["ʂai"],
    "shan": ["ʂan"],
    "shang": ["ʂɑŋ"],
    "shao": ["ʂaʌ"],
    "she": ["ʂø"],
    "shei": ["ʂei"],
    "shen": ["ʂœn"],
    "sheng": ["ʂɵŋ"],
    "shi": ["ʂʏ"],
    "shou": ["ʂou"],
    "shu": ["ʂu"],
    "shua": ["ʂua"],
    "shuai": ["ʂuai"],
    "shuan": ["ʂuan"],
    "shuang": ["ʂuɑŋ"],
    "shui": ["ʂuei"],
    "shun": ["ʂun"],
    "shuo": ["ʂuo"],
    "si": ["sɪ"],
    "song": ["soŋ"],
    "sou": ["sou"],
    "su": ["su"],
    "suan": ["suan"],
    "sui": ["suei"],
    "sun": ["sun"],
    "suo": ["suo"],
    "ta": ["ta"],
    "tai": ["tai"],
    "tan": ["tan"],
    "tang": ["tɑŋ"],
    "tao": ["taʌ"],
    "te": ["tø"],
    "tei": ["tei"],
    "teng": ["tɵŋ"],
    "ti": ["ti"],
    "tian": ["tiɛn"],
    "tiao": ["tiaʌ"],
    "tie": ["tie"],
    "ting": ["tɨŋ"],
    "tong": ["toŋ"],
    "tou": ["tou"],
    "tu": ["tu"],
    "tuan": ["tuan"],
    "tui": ["tuei"],
    "tun": ["tun"],
    "tuo": ["tuo"],
    "wa": ["wa"],
    "wai": ["wai"],
    "wan": ["wan"],
    "wang": ["wɑŋ"],
    "wei": ["wei"],
    "wen": ["wœn"],
    "weng": ["wɵŋ"],
    "wo": ["wo"],
    "wu": ["wu"],
    "xi": ["ɕi"],
    "xia": ["ɕia"],
    "xian": ["ɕiɛn"],
    "xiang": ["ɕiɑŋ"],
    "xiao": ["ɕiaʌ"],
    "xie": ["ɕie"],
    "xin": ["ɕin"],
    "xing": ["ɕɨŋ"],
    "xiong": ["ɕioŋ"],
    "xiu": ["ɕio"],
    "xu": ["ɕy"],
    "xuan": ["ɕyɛn"],
    "xue": ["ɕye"],
    "xun": ["ɕyn"],
    "ya": ["ia"],
    "yan": ["iɛn"],
    "yang": ["iɑŋ"],
    "yao": ["iaʌ"],
    "ye": ["ie"],
    "yi": ["i"],
    "yin": ["in"],
    "ying": ["ɨŋ"],
    "yo": ["io"],
    "yong": ["ioŋ"],
    "you": ["io"],
    "yu": ["y"],
    "yuan": ["yɛn"],
    "yue": ["ye"],
    "yun": ["yn"],
    "za": ["dza"],
    "zai": ["dzai"],
    "zan": ["dzan"],
    "zang": ["dzɑŋ"],
    "zao": ["dzaʌ"],
    "ze": ["dzø"],
    "zei": ["dzei"],
    "zen": ["dzœn"],
    "zeng": ["dzɵŋ"],
    "zha": ["dʒa"],
    "zhai": ["dʒai"],
    "zhan": ["dʒan"],
    "zhang": ["dʒɑŋ"],
    "zhao": ["dʒaʌ"],
    "zhe": ["dʒø"],
    # "zhei": ["dʒei"], it doesn't exist
    "zhen": ["dʒœn"],
    "zheng": ["dʒɵŋ"],
    "zhi": ["dʒʏ"],
    "zhong": ["dʒoŋ"],
    "zhou": ["dʒou"],
    "zhu": ["dʒu"],
    "zhua": ["dʒua"],
    "zhuai": ["dʒuai"],
    "zhuan": ["dʒuan"],
    "zhuang": ["dʒuɑŋ"],
    "zhui": ["dʒuei"],
    "zhun": ["dʒun"],
    "zhuo": ["dʒuo"],
    "zi": ["dzɪ"],
    "zong": ["dzoŋ"],
    "zou": ["dzou"],
    "zu": ["dzu"],
    "zuan": ["dzuan"],
    "zui": ["dzuei"],
    "zun": ["dzun"],
    "zuo": ["dzuo"],
}


def _chinese_character_to_pinyin(text: str) -> List[str]:
    pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)
    pinyins_flat_list = [item for sublist in pinyins for item in sublist]
    return pinyins_flat_list


def _chinese_pinyin_to_phoneme(pinyin: str) -> str:
    segment = pinyin[:-1]
    tone = pinyin[-1]
    phoneme = PINYIN_DICT.get(segment, [""])[0]
    return phoneme + tone

def chinese_text_to_phonemes(text: str) -> str:
    tokenized_text = jieba.cut(text, HMM=False)
    tokenized_text = " ".join(tokenized_text)
    pinyined_text: List[str] = _chinese_character_to_pinyin(tokenized_text)

    results: List[str] = []
    for token in pinyined_text:
        if token[-1] in "12345":  # TODO transform to is_pinyin()
            pinyin_phonemes = _chinese_pinyin_to_phoneme(token)

            results += list(pinyin_phonemes)
        else:  # is ponctuation or other
            results += list(token)

    return ''.join(results)

if __name__ == "__main__":
    text = chinese_text_to_phonemes("我们都是中国人，我爱家乡")
    print(text)