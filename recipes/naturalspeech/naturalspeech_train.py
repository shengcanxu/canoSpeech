import os
from trainer import Trainer, TrainerArgs

from config.config import NaturalSpeechConfig
from dataset.basic_dataset import get_metas_from_filelist
from recipes.naturalspeech.naturalspeech import NaturalSpeechTrain
from recipes.vits.vits import VitsTrain

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))
# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None
# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 8

def main():
    config = NaturalSpeechConfig(
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )
    config.load_json("./config/naturalspeech_vctk.json")
    # config.load_json("./config/naturalspeech_ljs.json")
    datasets = config.dataset_config.datasets
    # print(config)

    train_samples = get_metas_from_filelist([d.meta_file_train for d in datasets])
    test_samples = get_metas_from_filelist([d.meta_file_val for d in datasets])

    # init the model
    train_model = NaturalSpeechTrain(config=config)

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