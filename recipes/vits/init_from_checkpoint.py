from collections import OrderedDict
import fsspec
import torch
import json
from config.config import VitsConfig
from dataset.basic_dataset import get_metas_from_filelist
from recipes.vits.vits import VitsTrain
from trainer import Trainer, TrainerArgs

if __name__ == "__main__":
    pre_path = "D:\\project\\canoSpeech\\output\\online\\NaturalTTSLJSpeech-September-27-2023_VAE_train\\best_model.pth"
    with fsspec.open(pre_path, "rb") as f:
        pre_checkpoint = torch.load(f, map_location="cpu")
        pre_model = pre_checkpoint["model"]
    path = "D:\\dataset\\LJSpeech\\vits_pretrained_ljs.pth"
    with fsspec.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
        model = checkpoint["model"]

    # change the key in model
    path = "D:\\project\\canoSpeech\\recipes\\vits\\ljs_key_mapping.txt"
    with open(path, "r") as f:
        jsonstr = f.read()
        jsonobj = json.loads(jsonstr)

    new_model = pre_model
    for key in jsonobj:
        value = jsonobj[key]
        new_model[key] = model[value]

    config = VitsConfig()
    config.load_json("../../config/vits_ljspeech.json")
    data_config = config.dataset_config

    # init the model
    train_model = VitsTrain(config=config)
    train_model.load_state_dict(new_model, strict=False)

     # init the trainer and train
    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)
    trainer = Trainer(
        TrainerArgs(continue_path=config.continue_path, restore_path=config.restore_path),
        config,
        output_path=config.output_path,
        model=train_model,
        train_samples=train_samples,
        eval_samples=test_samples,
    )
    trainer.fit()
    print("done!")