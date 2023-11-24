""" from https://github.com/keithito/tacotron """
from dataset.baker import pinyin_dict
from text import cleaners
from text.symbols import symbols, symbols_cn


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_symbol_to_id_cn = {s: i for i, s in enumerate(symbols_cn)}
_id_to_symbol_cn = {i: s for i, s in enumerate(symbols_cn)}

# use to add_blank
def _intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def text_to_tokens(text, cleaner_names=["english_cleaners2"]):
    """Converts a string of text to a tokens of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a tokens
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    tokens = []

    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        tokens += [symbol_id]
    return tokens

def text_to_tokens_cn(text):
    tokens = []
    new_pinyins = []

    pinyins = text.split()
    for pinyin in pinyins:
        if pinyin == "sp":
            new_pinyins += ["sp"]
            continue

        ton = pinyin[-1]
        pinyin = pinyin[0:-1]
        sheng_mu, yun_mu = pinyin_dict[pinyin]
        new_pinyins += [sheng_mu, yun_mu + ton, "#0"]
    new_pinyins += ["eos"]

    for symbol in new_pinyins:
        symbol_id = _symbol_to_id_cn[symbol]
        tokens += [symbol_id]
    return tokens

def cleaned_text_to_tokens(cleaned_text, lang="en"):
    """Converts a string of text to a tokens of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a tokens
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if lang == "zh":
        tokens = [_symbol_to_id_cn[symbol] for symbol in cleaned_text.split()]
    else:
        tokens = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return tokens


def tokens_to_text(sequence, lang="en"):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol_cn[symbol_id] if lang == "zh" else _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text
