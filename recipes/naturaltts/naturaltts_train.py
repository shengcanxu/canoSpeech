import os
from dataset.share_duration import ShareDuration
from torch.cuda import memory_allocated
from trainer import Trainer, TrainerArgs
from config.config import NaturalTTSConfig
from dataset.dataset import get_metas_from_filelist
from recipes.naturaltts.naturaltts import NaturalTTSTrain


# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

def main():
    config = NaturalTTSConfig()
    config.load_json("./config/naturaltts_vctk.json")
    # config.load_json("./config/naturaltts_ljs.json")
    data_config = config.dataset_config
    # print(config)

    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    durations = ShareDuration()
    train_model = NaturalTTSTrain(config=config, share_vars=durations)

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