import argparse
import os
from multiprocessing import Pool
import text
from config.config import VitsConfig
from dataset.VCTK import load_vctk_metas as load_vctk_metas
from dataset.aishell3 import load_aishell3_metas
from dataset.baker import load_baker_metas
from dataset.cmltts import load_cmlpt_metas
from dataset.kokoro import load_kokoro_metas
from dataset.libritts import load_libritts_metas
from dataset.ljspeech import load_ljspeech_metas
from text import symbols
import numpy as np
from collections import Counter

"""
generate train and test filelist. 
"""
def load_file_metas(config):
    metas = []
    for dataset in config.datasets:
        dataset_name = dataset.dataset_name
        dataset_path = dataset.path
        if dataset_name.lower() == "vctk":
            items = load_vctk_metas(root_path=dataset_path, ignored_speakers=config.ignored_speakers)
            metas.extend(items)
        elif dataset_name.lower() == "ljspeech":
            items = load_ljspeech_metas(root_path=dataset_path)
            metas.extend(items)
        elif dataset_name.lower() == "baker":
            items = load_baker_metas(root_path=dataset_path)
            metas.extend(items)
        elif dataset_name.lower() == "libritts":
            items = load_libritts_metas(root_path=dataset_path)
            metas.extend(items)
        elif dataset_name.lower() == "cmlpt":
            items = load_cmlpt_metas(root_path=dataset_path)
            metas.extend(items)
        elif dataset_name.lower() == "kokoro":
            items = load_kokoro_metas(root_path=dataset_path)
            metas.extend(items)
        elif dataset_name.lower() == "aishell":
            items = load_aishell3_metas(root_path=dataset_path)
            metas.extend(items)
    return metas

def split_dataset_metas(items, eval_split_max_size=None, eval_split_size=0.01):
    """Split a dataset into train and eval. Consider speaker distribution in multi-speaker training.
    Args:
        items (List[List]):
            A list of samples. Each sample is a list of `[audio_path, text, speaker_id]`.
        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).
        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).
    :return:
        eval_datas, train_datas
    """
    speakers = [item["speaker"] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    if eval_split_size > 1:
        eval_split_size = int(eval_split_size)
    else:
        if eval_split_max_size:
            eval_split_size = min(eval_split_max_size, int(len(items) * eval_split_size))
        else:
            eval_split_size = int(len(items) * eval_split_size)

    assert ( eval_split_size > 0
    ), " [!] You do not have enough samples for the evaluation set. You can work` around this setting the 'eval_split_size' parameter to a minimum of {}".format(1 / len(items) )

    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item["speaker"] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx]["speaker"]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]

def gen_filelist(config_path:str):
    config = VitsConfig()
    config.load_json(config_path)
    datasets = config.dataset_config.datasets
    text_config = config.text

    for dataset_config in datasets:
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
    ptext, text_cleaner, lang = args
    original_text = ptext[3]
    cleaned_text = text.get_clean_text(original_text, text_cleaner)
    print(cleaned_text)
    return cleaned_text

# clean text and save to filelist file
def _gen_filelist_cleaned(train_filelist:str, test_filelist:str, text_cleaners, lang="en"):
    text_cleaner = text_cleaners.get(lang)
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
            clean_args = list(zip(texts, [text_cleaner] * len(texts), [lang] * len(texts)))
            with Pool(processes=10) as p:
                cleaned_texts = p.map(do_clean_text, clean_args)
                # cleaned_texts = do_clean_text(clean_args[0])

                with open(new_filelist, "a", encoding="utf-8") as fw:
                    for ptext, cleaned_text in zip(texts, cleaned_texts):
                        fw.writelines(ptext[0] + "|" + ptext[1] + "|" + ptext[2] + "|" + cleaned_text + "\n")

def check_symbol_coverage(config_path):
    """test and check if all texts are in the range of symbols used for training."""
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

def gen_other_filelists(config_path):
    config = VitsConfig()
    config.load_json(config_path)
    dataset_config = config.dataset_config.datasets[0]
    paths = [
        dataset_config.meta_file_train,
        dataset_config.meta_file_val,
        dataset_config.meta_file_train.replace(".cleaned", ""),
        dataset_config.meta_file_val.replace(".cleaned", "")
    ]

    datas_list = []
    for path in paths:
        datas = []
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                datas.append(line)
        datas_list.append(datas)
    print("load data done!")

    # all in windows
    win_all_paths = [path.replace("filelists", "filelists/all_backup") for path in paths]
    for path, datas in zip(win_all_paths, datas_list):
        with open(path, "w", encoding="utf-8") as fp:
            for line in datas:
                fp.write(line)

    # test in windows
    win_test_paths = [path.replace("filelists", "filelists/test") for path in paths]
    for path, datas in zip(win_test_paths, datas_list):
        with open(path, "w", encoding="utf-8") as fp:
            for line in datas[0:16]:
                fp.write(line)

    # all in linux
    linux_all_paths = [path.replace("filelists", "filelists/linux/all") for path in paths]
    for path, datas in zip(linux_all_paths, datas_list):
        with open(path, "w", encoding="utf-8") as fp:
            for line in datas:
                line = line.replace("D:/dataset", "/home/cano/dataset")
                fp.write(line)

    # test in linux
    linux_test_paths = [path.replace("filelists", "filelists/linux/test") for path in paths]
    for path, datas in zip(linux_test_paths, datas_list):
        with open(path, "w", encoding="utf-8") as fp:
            for line in datas[0:16]:
                line = line.replace("D:/dataset", "/home/cano/dataset")
                fp.write(line)

    print("All Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/vits_aishell.json")
    args = parser.parse_args()

    # gen_filelist(args.config)
    # check_symbol_coverage(args.config)
    gen_other_filelists(args.config)