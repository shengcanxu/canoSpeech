import argparse
import text
from config.config import VitsConfig
from dataset.VCTK import load_vctk_metas as load_vctk_metas
from dataset.dataset import split_dataset_metas
from dataset.ljspeech import load_ljspeech_metas

"""
generate train and test filelist. 
"""

def load_file_metas(config):
    dataset_name = config.dataset_name
    items = []
    if dataset_name.lower() == "vctk":
        items = load_vctk_metas(root_path=config.path, ignored_speakers=config.ignored_speakers)
    if dataset_name.lower() == "ljspeech":
        items = load_ljspeech_metas(root_path=config.path)
    return items

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/base.json")
    args = parser.parse_args()

    train_config = VitsConfig()
    train_config.load_json("../config/base.json")
    config = train_config.dataset_config
    text_config = train_config.text

    # load dataset metas
    data_items = load_file_metas(config)
    # split train and eval dataset
    test_datas, train_datas = split_dataset_metas(
        items=data_items,
        eval_split_max_size=train_config.eval_split_max_size,
        eval_split_size=train_config.eval_split_size
    )

    # save test and validate filelist
    filelists = [
        "../filelists/%s_train_filelist.txt" % config.dataset_name,
        "../filelists/%s_test_filelist.txt" % config.dataset_name
    ]
    with open(filelists[0], "w", encoding="utf-8") as f:
        f.writelines([x["audio"] + "|" + x["speaker"] + "|en|" + x["text"].strip() + "\n" for x in train_datas])
    with open(filelists[1], "w", encoding="utf-8") as f:
        f.writelines([x["audio"] + "|" + x["speaker"] + "|en|" + x["text"].strip() + "\n" for x in test_datas])

    # clean text and save to filelist file
    for filelist in filelists:
        print("Start clean:", filelist)
        with open(filelist, encoding="utf-8") as f:
            filepaths_and_text = [line.strip().split("|") for line in f]

            for i in range(len(filepaths_and_text)):
                original_text = filepaths_and_text[i][2]
                cleaned_text = text._clean_text(original_text, text_config.text_cleaners)
                filepaths_and_text[i][2] = cleaned_text

            new_filelist = filelist + ".cleaned"
            with open(new_filelist, "w", encoding="utf-8") as f:
                f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
