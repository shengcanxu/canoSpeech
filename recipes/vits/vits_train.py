import os
import torch
from trainer import Trainer, TrainerArgs
import torch.distributed as dist
# dataset path
from config.config import VitsConfig

# Name of the run for the Trainer
from dataset.dataset import get_metas_from_filelist
from recipes.vits.vits import VitsTrain

RUN_NAME = "CanoSpeech"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 8

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
# change the config dataset_config.sample_rate in config file
# SAMPLE_RATE = 16000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 10


def main():
    config = VitsConfig(
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )
    config.load_json("./config/vits.json")
    data_config = config.dataset_config
    # print(config)

    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    train_model = VitsTrain(config=config)

    # init the trainer and train
    trainer = Trainer(
        TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
        config,
        output_path=OUT_PATH,
        model=train_model,
        train_samples=train_samples,
        eval_samples=test_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()