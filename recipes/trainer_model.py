from trainer.torch import DistributedSampler
import platform
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Dataset
from trainer import TrainerModel, get_optimizer, get_scheduler

from util.helper import sequence_mask
from util.mel_processing import wav_to_spec, spec_to_mel, wav_to_mel
from dataset.basic_dataset import TextAudioDataset
from typing import Dict, List, Union, Tuple
from coqpit import Coqpit

class TrainerModelWithDataset(TrainerModel):
    def __init__(self, config: Coqpit) -> None:
        super().__init__()
        self.config = config

    def get_sampler(self, config: Coqpit, dataset, num_gpus=1, rank=0):
        if num_gpus == 1:
            return RandomSampler(dataset)
        else:
            return DistributedSampler(dataset, shuffle=True, rank=rank, num_replicas=num_gpus)

    def get_dataset(self, config: Coqpit, samples):
        return TextAudioDataset(samples, config)

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
            return loader

        dataset = self.get_dataset(config, samples)

        # wait all the DDP process to be ready
        if num_gpus > 1:
            dist.barrier()

        sampler = self.get_sampler(config, dataset, num_gpus, rank)

        # #TODO: fix this:
        #  set num_workers>0 the DataLoader will be very slow in windows, because it re-start
        # all processes every epoch. https://github.com/JaidedAI/EasyOCR/issues/274
        num_workers = config.dataset_config.num_eval_loader_workers if is_eval else config.dataset_config.num_loader_workers
        if platform.system() == "Windows":
            num_workers = 0
        loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
            # persistent_workers=True,
        )
        return loader

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device. datas are send to GPU before calling this func"""
        if self.config.dataset_config.melspec_use_GPU:
            wav = batch["waveform"]
            audio_config = self.config.audio
            spec = wav_to_spec(
                wav=wav,
                n_fft=audio_config.fft_size,
                hop_size=audio_config.hop_length,
                win_size=audio_config.win_length,
                center=False
            )
            batch["spec"] = spec

            mel = spec_to_mel(
                spec=spec,
                n_fft=audio_config.fft_size,
                num_mels=audio_config.num_mels,
                sample_rate=audio_config.sample_rate,
                fmin=audio_config.mel_fmin,
                fmax=audio_config.mel_fmax
            )
            batch["mel"] = mel

            # compute spectrogram frame lengths
            batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
            batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

            # zero the padding frames
            batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
            batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.
        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.
        Returns:
            List: Schedulers, one for each optimizer.
        """
        if isinstance(optimizer, list):
            schedulers = []
            for optim in optimizer:
                scheduler = get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optim)
                schedulers.append(scheduler)
            return schedulers
        else:
            scheduler = get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)
            return scheduler
