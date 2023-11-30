import argparse
import os
import pickle
import random
import time
from typing import Dict, Tuple, List

from speaker.speaker_encoder import SpeakerEncoder
from text import text_to_tokens, _intersperse
import torch
from torch import nn
from coqpit import Coqpit
from dataset.sampler import DistributedBucketSampler
from language.languages import LanguageManager
from layers.discriminator import VitsDiscriminator
from layers.losses import VitsDiscriminatorLoss, VitsGeneratorLoss
from recipes.trainer_model import TrainerModelWithDataset
from recipes.vits.vits import VitsModel
from torch.cuda.amp import autocast

from trainer import Trainer, TrainerArgs, get_optimizer
from config.config import VitsConfig
from dataset.basic_dataset import get_metas_from_filelist, TextAudioDataset
import soundfile as sf
from util.helper import segment
from util.mel_processing import wav_to_mel, load_audio


class VitsTrain(TrainerModelWithDataset):
    """
    VITS and YourTTS model training model.
    """
    def __init__(self, config:VitsConfig):
        super().__init__(config)
        self.config = config
        self.model_config = config.model
        self.balance_disc_generator = config.balance_disc_generator
        self.skip_discriminator = False

        self.generator = VitsModel(config=config)
        self.discriminator = VitsDiscriminator(
            periods=self.model_config.discriminator.periods_multi_period,
            use_spectral_norm=self.model_config.discriminator.use_spectral_norm,
        )

    def train_step(self, batch: Dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        spec_lens = batch["spec_lens"]
        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lens = batch["token_lens"]
            spec = batch["spec"]
            waveform = batch["waveform"]
            speaker_ids = batch["speaker_ids"]
            speaker_embeds = batch["speaker_embeds"]

            # generator pass
            outputs = self.generator(
                x=tokens,
                x_lengths=token_lens,
                y=spec,
                y_lengths=spec_lens,
                waveform=waveform,
                speaker_embeds=speaker_embeds,
                speaker_ids=speaker_ids,
                language_ids=None
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs

            # compute scores and features
            scores_disc_real, _, scores_disc_fake, _ = self.discriminator(
                x=outputs["waveform_seg"],
                x_hat=outputs["y_hat"].detach(),
            )

            # compute loss
            with autocast(enabled=False):
                loss_dict = criterion[0](
                    scores_disc_real=scores_disc_real,
                    scores_disc_fake=scores_disc_fake,
                )

            self.disc_loss_dict = loss_dict
            if self.skip_discriminator:
                return outputs, None
            else:
                return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment
            with autocast(enabled=False):
                mel_slice = segment(
                    x=mel.float(),
                    segment_indices=self.model_outputs_cache["slice_ids"],
                    segment_size=self.model_config.spec_segment_size,
                    pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["y_hat"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            _, feats_disc_real, scores_disc_fake, feats_disc_fake = self.discriminator(
                x=self.model_outputs_cache["waveform_seg"],
                x_hat=self.model_outputs_cache["y_hat"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[1](
                    mel_slice=mel_slice_hat.float(),
                    mel_slice_hat=mel_slice.float(),
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=spec_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    use_speaker_encoder_as_loss=self.model_config.use_speaker_encoder_as_loss,
                    gt_speaker_emb=None,
                    syn_speaker_emb=None,
                )

                if self.balance_disc_generator:
                    loss_dict["loss_disc"] = self.disc_loss_dict["loss_disc"]
                    loss_dict["loss_disc_real_all"] = self.disc_loss_dict["loss_disc_real_all"]
                    loss_dict["loss_disc_fake_all"] = self.disc_loss_dict["loss_disc_fake_all"]
                    # auto balance discriminator and generator, make sure loss of disciminator will be roughly 1.5x - 2.0x of generator
                    self.skip_discriminator = loss_dict["loss_disc"] < 0.4 * self.config.loss.disc_loss_alpha

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in train_step()`"""
        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters."""
        # select generator parameters
        discOptimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr[0],
            model=self.discriminator
        )

        genOptimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr[1],
            model=self.generator
        )
        return [discOptimizer, genOptimizer]

    def get_sampler(self, config: Coqpit, dataset, num_gpus=1, rank=0):
        return DistributedBucketSampler(
            dataset=dataset,
            batch_size=config.batch_size,
            boundaries=[32,300,400,500,600,700,800,900,1000],
            num_replicas=num_gpus,
            rank=0,
            shuffle=True
        )

    def get_dataset(self, config: Coqpit, samples):
        return TextAudioDataset(samples, config)

    def forward(self, input: torch.Tensor) -> Dict:
        print("nothing to do! doing the real train code in train_step. ")
        return input

    def inference(self, text:str, speaker_id:int=None, speaker_embed=None):
        tokens = text_to_tokens(text)
        if self.config.text.add_blank:
            tokens = _intersperse(tokens, 0)
        tokens = torch.LongTensor(tokens).unsqueeze(dim=0).cuda()
        x_lengths = torch.LongTensor([tokens.size(1)]).cuda()

        speaker_ids = torch.LongTensor([speaker_id]).cuda() if speaker_id is not None else None
        speaker_embed = torch.FloatTensor(speaker_embed).unsqueeze(0).cuda() if speaker_embed is not None else None

        self.generator.eval()
        wav, _, _, _ = self.generator.infer(
            tokens,
            x_lengths,
            speaker_ids=speaker_ids,
            speaker_embeds=speaker_embed,
            noise_scale=0.8,
            length_scale=1,
        )
        return wav

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        output, loss_dict = self.train_step(batch, criterion, optimizer_idx)
        if optimizer_idx == 0:
            spec = batch["spec"][[0]]
            spec_len = batch["spec_lens"][[0]]
            speaker_id = batch["speaker_ids"][[0]]
            speaker_embed = batch["speaker_embeds"][[0]]
            wav = self.generator.generate_z_wav(spec, spec_len, speaker_id, speaker_embed)

            wav = wav[0, 0].cpu().float().numpy()
            filename = os.path.basename(batch["filenames"][0])
            sf.write(f"{self.config.output_path}/{int(time.time())}_{filename}", wav, 22050)

        return output, loss_dict

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        output_path = assets["output_path"]
        print("doing test run...")
        text = "Who else do you want to talk to? You can go with me today to the meeting."

        # speaker_id = random.randint(0, 9)
        # path1 = "D:/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav.pkl"
        # path2 = "D:/dataset/VCTK/wav48_silence_trimmed/p273/p273_004_mic1.flac.wav.pkl"
        path1 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p253/p253_003_mic1.flac.wav.pkl"
        path2 = "/home/cano/dataset/VCTK/wav48_silence_trimmed/p273/p273_004_mic1.flac.wav.pkl"
        path = path1 if random.randint(1,10) >= 5 else path2
        fp = open(path, "rb")
        pickleObj = pickle.load(fp)
        speaker_embed = pickleObj["speaker"]

        wav = self.inference(text, speaker_embed=speaker_embed)
        wav = wav[0, 0].cpu().float().numpy()
        sf.write(f"{output_path}/test_{int(time.time())}.wav", wav, 22050)

def test(model, filepath:str):
    # speaker embedding
    wav, sr = load_audio(filepath)
    speaker_encoder = SpeakerEncoder(
        config_path="/home/cano/dataset/VCTK/config_se.json",
        model_path="/home/cano/dataset/VCTK/model_se.pth.tar",
        use_cuda=True
    )
    speaker_embed = speaker_encoder.compute_embedding_from_waveform(wav)
    speaker_embed = speaker_embed.squeeze(0)
    speaker_embed = speaker_embed.cpu().float().numpy()

    text = "I am a student but I don't want to do any homework."
    wav = model.inference(text, speaker_embed=speaker_embed)
    wav = wav[0, 0].cpu().float().numpy()
    sf.write(f"{filepath}.test.wav", wav, 22050)

def main(config_path:str):
    config = VitsConfig()
    config.load_json(config_path)
    data_config = config.dataset_config

    train_samples = get_metas_from_filelist(data_config.meta_file_train)
    test_samples = get_metas_from_filelist(data_config.meta_file_val)

    # init the model
    train_model = VitsTrain(config=config)

    # init the trainer and train
    trainer = Trainer(
        TrainerArgs(continue_path=config.continue_path, restore_path=config.restore_path, skip_train_epoch=False),
        config,
        output_path=config.output_path,
        model=train_model,
        train_samples=train_samples,
        eval_samples=test_samples,
    )
    trainer.fit()

    # test(train_model, "/home/cano/output/test/test.wav")
    # test(train_model, "/home/cano/output/test/test2.wav")
    # test(train_model, "/home/cano/output/test/test3.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vits vctk train", formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("--config_path", type=str, default="./config/vits_vctk.json", required=False)
    args = parser.parse_args()

    main(args.config_path)