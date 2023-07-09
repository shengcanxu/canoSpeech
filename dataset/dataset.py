import os
import random
import numpy as np
from collections import Counter

import torchaudio

from text import text_to_sequence, cleaned_text_to_sequence, _clean_text
import torch
from torch.utils.data import Dataset, DistributedSampler


def split_dataset_metas(items, eval_split_max_size=None, eval_split_size=0.01):
    """Split a dataset into train and eval. Consider speaker distribution in multi-speaker training.
    Args:
        items (List[List]):
            A list of samples. Each sample is a list of `[audio_path, text, speaker_id]`.
        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).
        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).
    :return:
        eval_datas, train_datas
    """
    speakers = [item["speaker"] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    if eval_split_size > 1:
        eval_split_size = int(eval_split_size)
    else:
        if eval_split_max_size:
            eval_split_size = min(eval_split_max_size, int(len(items) * eval_split_size))
        else:
            eval_split_size = int(len(items) * eval_split_size)

    assert ( eval_split_size > 0
    ), " [!] You do not have enough samples for the evaluation set. You can work around this setting the 'eval_split_size' parameter to a minimum of {}".format(1 / len(items) )

    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item["speaker"] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx]["speaker"]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]


def get_metas_from_filelist(filelist:str):
    """get meta from filelist, filelist is generated from preprocess.py"""
    metas = []
    with open(filelist, encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("|")
            metas.append({
                "text": items[2],
                "speaker": items[1],
                "audio": items[0],
            })
    return metas

class TextAudioDataset(Dataset):
    """
    load audio, text and return dataset
    """
    def __init__(self, samples, config):
        self.samples = samples

        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.sample_rate = config.sample_rate

        self.cleaned_text = getattr(config, "cleaned_text", False)
        self.text_cleaners = config.text_cleaners
        self.add_blank = config.add_blank
        self.min_text_len = getattr(config, "min_text_len", 1)
        self.max_text_len = getattr(config, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.samples)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        samples_new = []
        lengths = []
        for sample in self.samples:
            if self.min_text_len <= len(sample["text"]) <= self.max_text_len:
                samples_new.append(sample)
                lengths.append(os.path.getsize(sample["audio"]) // (2 * self.hop_length))
        self.samples = samples_new
        self.lengths = lengths

    def __getitem__(self, index):
        return self.get_audio_text_duration_f0(self.samples[index])

    def __len__(self):
        return len(self.samples)

    def get_audio_text_duration_f0(self, sample):
        tokens, phoneme = self._get_text(sample["text"])
        wav = self._get_audio(sample["audio"])

        #TODO:change code here
        duration = torch.zeros_like(tokens)
        f0 = torch.zeros_like(tokens)
        return {
            "raw_text": sample["text"],
            "phoneme": phoneme,
            "tokens": tokens,
            "token_len": len(tokens),
            "wav": wav,
            "audio_file": sample["audio"],
            "speaker": sample["speaker"],
            "duration": duration,
            "f0": f0
        }

    def _get_audio(self, filename):
        wav, sample_rate = torchaudio.load(filename)
        if sample_rate != self.sample_rate:
            raise ValueError( "{} SR doesn't match target {} SR".format( sample_rate, self.sample_rate) )
        return wav

    def _get_text(self, text):
        """format text and add blank"""
        if self.cleaned_text:
            cleaned_text = text
            tokens = cleaned_text_to_sequence(cleaned_text)
        else:
            cleaned_text = _clean_text(text, self.text_cleaners)
            tokens = cleaned_text_to_sequence(cleaned_text)
        if self.add_blank:
            tokens = self._intersperse(tokens, 0)
        tokens = torch.LongTensor(tokens)
        return tokens, cleaned_text

    @staticmethod
    def _intersperse(lst, item):
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result


class DistributedBucketSampler(DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True, ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)])

            # subsample
            ids_bucket = ids_bucket[self.rank:: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size: (j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


class TextAudioCollate:
    """Zero-pads model inputs, audios and targets and pad a batch"""

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speakers: :math:`[B]`
            - audio_files: :math:`[B]`
            - raw_texts: :math:`[B]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch["wav"]]), dim=0, descending=True
        )

        max_text_len = max([len(x) for x in batch["tokens"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()
        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        token_padded = token_padded.zero_()
        wav_padded = wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            tokens = batch["tokens"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(tokens)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,
            "waveform_lens": wav_lens,
            "waveform_rel_lens": wav_rel_lens,
            "speakers": batch["speaker"],
            "audio_files": batch["audio_file"],
            "raw_texts": batch["raw_text"]
        }
