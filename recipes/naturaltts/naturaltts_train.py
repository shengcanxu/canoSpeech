import os
from trainer import Trainer, TrainerArgs
from config.config import NaturalTTSConfig
from dataset.dataset import get_metas_from_filelist
from recipes.naturaltts.naturaltts import NaturalTTSTrain

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None
# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 8

def main():
    config = NaturalTTSConfig(
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )
    config.load_json("./config/naturaltts_vctk.json")
    # config.load_json("./config/naturaltts_ljs.json")
    data_config = config.dataset_config
    # print(config)

    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    train_model = NaturalTTSTrain(config=config)

    # init the trainer and train
    trainer = Trainer(
        TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
        config,
        output_path=config.output_path,
        model=train_model,
        train_samples=train_samples,
        eval_samples=test_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()