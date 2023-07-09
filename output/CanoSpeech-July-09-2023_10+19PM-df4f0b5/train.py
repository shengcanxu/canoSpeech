import os
import torch
from trainer import Trainer, TrainerArgs, TrainerConfig
import torch.distributed as dist
# dataset path
from config.config import TTSDatasetConfig, TrainTTSConfig

# Name of the run for the Trainer
from dataset.VCTK import load_file_metas
from dataset.dataset import split_dataset_metas, get_metas_from_filelist
from model import SpeechModel

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
    # public environment
    rank = 0
    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "63331"
    dist.init_process_group(
        # backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank  # use gloo in windows
    )

    config = TrainTTSConfig(
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )
    config.load_json("./config/train.json")
    data_config = config.dataset_config
    print(config)

    train_datas = get_metas_from_filelist(data_config.meta_file_train)
    test_datas = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    model = SpeechModel(config=config)

    # init the trainer and train
    trainer = Trainer(
        TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_datas,
        eval_samples=test_datas,
    )
    trainer.fit()


if __name__ == "__main__":
    main()