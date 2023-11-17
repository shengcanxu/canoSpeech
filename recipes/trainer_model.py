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
