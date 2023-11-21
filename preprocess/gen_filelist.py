import argparse
import os

import text
from config.config import VitsConfig
from dataset.VCTK import load_vctk_metas as load_vctk_metas
from dataset.baker import load_baker_metas
from dataset.basic_dataset import split_dataset_metas
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
    if dataset_name.lower() == "baker":
        items = load_baker_metas(root_path=config.path)
    return items

def gen_filelist(config_path:str):
    config = VitsConfig()
    config.load_json(config_path)
    dataset_config = config.dataset_config
    text_config = config.text
    language = dataset_config.language

    # save test and validate filelist
    train_filelist = "../filelists/%s_train_filelist.txt" % dataset_config.dataset_name
    test_filelist = "../filelists/%s_test_filelist.txt" % dataset_config.dataset_name

    # load dataset metas
    print("load and split metas....")
    data_items = load_file_metas(dataset_config)
    # split train and eval dataset
    test_datas, train_datas = split_dataset_metas(
        items=data_items,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size
    )

    print("generate filelist....")
    if not os.path.exists(train_filelist):
        with open(train_filelist, "w", encoding="utf-8") as f:
            f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["text"].strip() + "\n" for x in train_datas])
    if not os.path.exists(test_filelist):
        with open(test_filelist, "w", encoding="utf-8") as f:
            f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["text"].strip() + "\n" for x in test_datas])

    if language == "en":
        print("generate english cleaned filelist....")
        _gen_filelist_cleaned(train_filelist, test_filelist, text_config.text_cleaners)
    elif language == "zh":
        print("generate chinese cleaned filelist....")
        _write_filelist_cleaned(train_filelist, train_datas, test_filelist, test_datas)

# create cleaned filelist from original filelist
def _write_filelist_cleaned(train_filelist:str, train_datas:list, test_filelist:str, test_datas:list):
    train_filelist_cleaned = train_filelist + ".cleaned"
    if not os.path.exists(train_filelist_cleaned):
        with open(train_filelist_cleaned, "w", encoding="utf-8") as f:
            f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["pinyin"].strip() + "\n" for x in train_datas])
    test_filelist_cleaned = test_filelist + ".cleaned"
    if not os.path.exists(test_filelist_cleaned):
        with open(test_filelist_cleaned, "w", encoding="utf-8") as f:
            f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["pinyin"].strip() + "\n" for x in test_datas])


# clean text and save to filelist file
def _gen_filelist_cleaned(train_filelist:str, test_filelist:str, text_cleaners):
    for filelist in [train_filelist, test_filelist]:
        print("Start clean:", filelist)
        with open(filelist, encoding="utf-8") as f:
            ptexts = [line.strip().split("|") for line in f]

            new_filelist = filelist + ".cleaned"
            fw = open(new_filelist, "a", encoding="utf-8")
            with open(new_filelist, encoding="utf-8") as fr:
                parsed_lines = len([line for line in fr])
                print(f"already parsed {parsed_lines} lines")

            for i in range(len(ptexts)):
                if i < parsed_lines: continue

                original_text = ptexts[i][3]
                cleaned_text = text._clean_text(original_text, text_cleaners)
                fw.writelines(ptexts[i][0] + "|" + ptexts[i][1] + "|" + ptexts[i][2] + "|" + cleaned_text + "\n")
                if i % 100 == 0:
                    print(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="../config/naturaltts_ljspeech.json")
    parser.add_argument("--config", type=str, default="../config/vits_baker.json")
    args = parser.parse_args()

    gen_filelist(args.config)