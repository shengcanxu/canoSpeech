from config.config import VitsConfig
from coqpit import Coqpit
from text import symbols, get_clean_text
from text.symbols import zh_symbols, ja_symbols


class SymbolManager():
    def __init__(self, dataset_config:Coqpit):
        self.symbols_map = {
            "en": {
                "prefix": -1,
                "length": 190,
                "symbol_to_id": {s: i for i, s in enumerate(symbols)},
                "id_to_symbol": {i: s for i, s in enumerate(symbols)}
            },
            "pt": {
                "prefix": -1,
                "length": 190,
                "symbol_to_id": {s: i for i, s in enumerate(symbols)},
                "id_to_symbol": {i: s for i, s in enumerate(symbols)}
            },
            "zh": {
                "prefix": -1,
                "length": 70,
                "symbol_to_id": {s: i for i, s in enumerate(zh_symbols)},
                "id_to_symbol": {i: s for i, s in enumerate(zh_symbols)}
            },
            "ja": {
                "prefix": -1,
                "length": 50,
                "symbol_to_id": {s: i for i, s in enumerate(ja_symbols)},
                "id_to_symbol": {i: s for i, s in enumerate(ja_symbols)}
            }
        }

        datasets = dataset_config.datasets
        self.languages = [dataset.language for dataset in datasets]
        self.languages.sort()
        idx = 0
        for lang in self.languages:
            lang_obj = self.symbols_map.get(lang, {})
            lang_obj["prefix"] = idx
            idx += lang_obj["length"]

    def symbol_count(self):
        lengths = [v["length"] for k, v in self.symbols_map.items() if v["prefix"] != -1]
        return sum(lengths) + 1

    def _symbol_to_id(self, symbol:str, lang="en"):
        lang_obj = self.symbols_map.get(lang)
        if lang_obj is None or lang_obj["prefix"] == -1: return 0

        prefix = lang_obj["prefix"]
        idx = lang_obj["symbol_to_id"].get(symbol, 0)
        return prefix + idx

    def _id_to_symbol(self, id:int, lang="en"):
        lang_obj = self.symbols_map.get(lang)
        if lang_obj is None or lang_obj["prefix"] == -1: return None

        prefix = lang_obj["prefix"]
        idx = id - prefix
        symbol = lang_obj["id_to_symbol"].get(idx, " ")
        return symbol

    def text_to_tokens(self, text, cleaner_name="english_cleaners2", lang="en"):
        """Converts a string of text to a tokens of IDs corresponding to the symbols in the text.
        Args:
          text: string to convert to a tokens
          cleaner_name: names of the cleaner functions to run the text through
        Returns:
          List of integers corresponding to the symbols in the text
        """
        clean_text = get_clean_text(text, cleaner_name)
        return self.cleaned_text_to_tokens(clean_text, lang=lang)

    def cleaned_text_to_tokens(self, cleaned_text, lang="en"):
        """Converts a string of text to a tokens of IDs corresponding to the symbols in the text.
        Args:
          text: string to convert to a tokens
        Returns:
          List of integers corresponding to the symbols in the text
        """
        tokens = [self._symbol_to_id(symbol, lang) for symbol in cleaned_text]
        return tokens

    def tokens_to_text(self, sequence, lang="en"):
        """Converts a sequence of IDs back to a string"""
        result = ""
        for symbol_id in sequence:
            s = self._id_to_symbol(symbol_id, lang)
            result += s
        return result


if __name__ == "__main__":
    config_path = "../config/vits_multilang.json"
    # config_path = "../config/vits_vctk.json"
    config = VitsConfig()
    config.load_json(config_path)
    manager = SymbolManager(config.dataset_config)

    idx = manager._symbol_to_id("e", lang="en")
    print(idx)
    idx = manager._symbol_to_id("e", lang="pt")
    print(idx)
    idx = manager._symbol_to_id("e", lang="zh")
    print(idx)
    idx = manager._symbol_to_id("e", lang="ja")
    print(idx)