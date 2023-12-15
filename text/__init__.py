""" from https://github.com/keithito/tacotron """
from dataset.baker import pinyin_dict
from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# use to add_blank
def _intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def text_to_tokens(text, cleaner_name="english_cleaners2"):
    """Converts a string of text to a tokens of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a tokens
      cleaner_name: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    clean_text = get_clean_text(text, cleaner_name)
    return cleaned_text_to_tokens(clean_text)


def cleaned_text_to_tokens(cleaned_text):
    """Converts a string of text to a tokens of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a tokens
    Returns:
      List of integers corresponding to the symbols in the text
    """
    tokens = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return tokens


def tokens_to_text(sequence, lang="en"):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def get_clean_text(text, cleaner_name):
    cleaner = getattr(cleaners, cleaner_name)
    if not cleaner:
        raise Exception("Unknown cleaner: %s" % cleaner_name)
    text = cleaner(text)
    return text
