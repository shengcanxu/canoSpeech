from coqpit import Coqpit


class LanguageManager():
    def __init__(self, dataset_config:Coqpit):
        datasets = dataset_config.datasets
        languages = set()
        for dataset in datasets:
            languages.add(dataset.language)

        languages = list(languages)
        languages.sort()
        idx = 0
        self.language_ids_map = {}
        for lang in languages:
            idx += 1
            self.language_ids_map[lang] = idx

    def get_language_id(self, lang):
        return self.language_ids_map.get(lang, 0)

    def language_count(self):
        return len(self.language_ids_map) + 1