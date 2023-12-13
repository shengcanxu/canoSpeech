import argparse
import os
from multiprocessing import Pool
import text
from config.config import VitsConfig
from dataset.VCTK import load_vctk_metas as load_vctk_metas
from dataset.baker import load_baker_metas
from dataset.basic_dataset import split_dataset_metas
from dataset.cmltts import load_cmlpt_metas
from dataset.kokoro import load_kokoro_metas
from dataset.libritts import load_libritts_metas
from dataset.ljspeech import load_ljspeech_metas
from text import symbols

"""
generate train and test filelist. 
"""

def load_file_metas(config):
    dataset_name = config.dataset_name
    items = []
    if dataset_name.lower() == "vctk":
        items = load_vctk_metas(root_path=config.path, ignored_speakers=config.ignored_speakers)
    elif dataset_name.lower() == "ljspeech":
        items = load_ljspeech_metas(root_path=config.path)
    elif dataset_name.lower() == "baker":
        items = load_baker_metas(root_path=config.path)
    elif dataset_name.lower() == "libritts":
        items = load_libritts_metas(root_path=config.path)
    elif dataset_name.lower() == "cmlpt":
        items = load_cmlpt_metas(root_path=config.path)
    elif dataset_name.lower() == "kokoro":
        items = load_kokoro_metas(root_path=config.path)
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

    print("generate filelist....")
    _gen_filelist(train_filelist, test_filelist, config)

    print("generate english cleaned filelist....")
    _gen_filelist_cleaned(train_filelist, test_filelist, text_config.text_cleaners, lang=language)


def _gen_filelist(train_filelist:str, test_filelist:str, config):
    if not os.path.exists(train_filelist) and not os.path.exists(test_filelist):
        # load dataset metas
        print("load and split metas....")
        data_items = load_file_metas(config.dataset_config)
        # split train and eval dataset
        test_datas, train_datas = split_dataset_metas(
            items=data_items,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size
        )

        with open(train_filelist, "w", encoding="utf-8") as f:
            f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["text"].strip() + "\n" for x in train_datas])
        with open(test_filelist, "w", encoding="utf-8") as f:
            f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["text"].strip() + "\n" for x in test_datas])

def do_clean_text(args):
    ptext, text_cleaners, lang = args
    original_text = ptext[3]
    cleaned_text = text._clean_text(original_text, text_cleaners)
    print(cleaned_text)
    return cleaned_text

# clean text and save to filelist file
def _gen_filelist_cleaned(train_filelist:str, test_filelist:str, text_cleaners, lang="en"):
    for filelist in [train_filelist, test_filelist]:
        print("Start clean:", filelist)
        ptexts = []
        with open(filelist, encoding="utf-8") as f:
            ptexts = [line.strip().split("|") for line in f]

        new_filelist = filelist + ".cleaned"
        parsed_lines = 0
        if os.path.exists(new_filelist):
            with open(new_filelist, encoding="utf-8") as fr:
                parsed_lines = len([line for line in fr])
                print(f"already parsed {parsed_lines} lines")

        pos = parsed_lines
        while pos < len(ptexts):
            print(pos)
            texts = ptexts[pos:pos+1000]
            pos += 1000
            clean_args = list(zip(texts, [text_cleaners] * len(texts), [lang] * len(texts)))
            with Pool(processes=10) as p:
                cleaned_texts = p.map(do_clean_text, clean_args)
                # cleaned_texts = do_clean_text(clean_args[0])

                with open(new_filelist, "a", encoding="utf-8") as fw:
                    for ptext, cleaned_text in zip(texts, cleaned_texts):
                        fw.writelines(ptext[0] + "|" + ptext[1] + "|" + ptext[2] + "|" + cleaned_text + "\n")


# test and check if all texts are in the range of symbols used for training.
def check_symbol_coverage(config_path):
    config = VitsConfig()
    config.load_json(config_path)
    train_filelist = config.dataset_config.meta_file_train
    test_filelist = config.dataset_config.meta_file_val
    for path in [train_filelist, test_filelist]:
        lines = []
        with open(path, "r", encoding="utf8") as fp:
            lines = fp.readlines()
        for line in lines:
            text = line.split("|")[3]
            text = text.strip()
            for t in text:
                if t not in symbols:
                    print("error on " + text + "  in " + t)
    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/vits_baker.json")
    args = parser.parse_args()

    # gen_filelist(args.config)
    check_symbol_coverage(args.config)