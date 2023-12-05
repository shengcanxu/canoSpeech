import argparse
import os
from multiprocessing import Pool
import text
from config.config import VitsConfig
from dataset.VCTK import load_vctk_metas as load_vctk_metas
from dataset.baker import load_baker_metas
from dataset.basic_dataset import split_dataset_metas
from dataset.libritts import load_libritts_metas
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
    if dataset_name.lower() == "libritts":
        items = load_libritts_metas(root_path=config.path)
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

    # # load dataset metas
    # print("load and split metas....")
    # data_items = load_file_metas(dataset_config)
    # # split train and eval dataset
    # test_datas, train_datas = split_dataset_metas(
    #     items=data_items,
    #     eval_split_max_size=config.eval_split_max_size,
    #     eval_split_size=config.eval_split_size
    # )
    #
    # print("generate filelist....")
    # if not os.path.exists(train_filelist):
    #     with open(train_filelist, "w", encoding="utf-8") as f:
    #         f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["text"].strip() + "\n" for x in train_datas])
    # if not os.path.exists(test_filelist):
    #     with open(test_filelist, "w", encoding="utf-8") as f:
    #         f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["text"].strip() + "\n" for x in test_datas])

    # if language == "en":
    print("generate english cleaned filelist....")
    _gen_filelist_cleaned(train_filelist, test_filelist, text_config.text_cleaners, lang=language)
    # elif language == "zh":
    #     print("generate chinese cleaned filelist....")
    #     _gen_filelist_cleaned_cn(train_filelist, train_datas, test_filelist, test_datas)

# # create cleaned filelist from original filelist
# def _gen_filelist_cleaned_cn(train_filelist:str, train_datas:list, test_filelist:str, test_datas:list):
#     train_filelist_cleaned = train_filelist + ".cleaned"
#     if not os.path.exists(train_filelist_cleaned):
#         with open(train_filelist_cleaned, "w", encoding="utf-8") as f:
#             f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["pinyin"].strip() + "\n" for x in train_datas])
#     test_filelist_cleaned = test_filelist + ".cleaned"
#     if not os.path.exists(test_filelist_cleaned):
#         with open(test_filelist_cleaned, "w", encoding="utf-8") as f:
#             f.writelines([x["audio"] + "|" + x["speaker"] + "|" + x["language"] + "|" + x["pinyin"].strip() + "\n" for x in test_datas])


def do_clean_text(args):
    ptext, text_cleaners, lang = args
    original_text = ptext[3]
    print(original_text)
    cleaned_text = text._clean_text(original_text, text_cleaners)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/vits_baker.json")
    args = parser.parse_args()

    gen_filelist(args.config)