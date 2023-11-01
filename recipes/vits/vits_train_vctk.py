import os
from trainer import Trainer, TrainerArgs
from config.config import VitsConfig
from dataset.basic_dataset import get_metas_from_filelist
from recipes.vits.vits import VitsTrain

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False


def main():
    config = VitsConfig()
    config.load_json("./config/vits_vctk.json")
    data_config = config.dataset_config

    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    train_model = VitsTrain(config=config)

    # init the trainer and train
    trainer = Trainer(
        TrainerArgs(continue_path=config.continue_path, restore_path=config.restore_path, skip_train_epoch=SKIP_TRAIN_EPOCH),
        config,
        output_path=config.output_path,
        model=train_model,
        train_samples=train_samples,
        eval_samples=test_samples,
    )
    trainer.fit()

if __name__ == "__main__":
    main()