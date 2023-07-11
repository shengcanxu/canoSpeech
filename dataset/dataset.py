import os
import random
import numpy as np
from collections import Counter
import torchaudio
import pickle
from text import cleaned_text_to_sequence, _clean_text
import torch
from torch.utils.data import Dataset

from util.audio_processor import AudioProcessor


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
        self.use_cache = getattr(config.dataset_config, "use_cache", False)
        self.add_spec = getattr(config.dataset_config, "add_spec", False)
        self.add_mel = getattr(config.dataset_config, "add_mel", False)

        self.hop_length = config.audio.hop_length
        self.win_length = config.audio.win_length
        self.sample_rate = config.audio.sample_rate

        self.processor = AudioProcessor(
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            sample_rate=config.audio.sample_rate,
            mel_fmin=config.audio.mel_fmin,
            mel_fmax=config.audio.mel_fmax,
            fft_size=config.audio.fft_length,
            num_mels=config.audio.num_mels,
            verbose=False
        )

        self.cleaned_text = getattr(config.text, "cleaned_text", False)
        self.text_cleaners = config.text.text_cleaners
        self.add_blank = config.text.add_blank
        self.min_text_len = getattr(config.text, "min_text_len", 1)
        self.max_text_len = getattr(config.text, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.samples)  # shuffle samples
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
        sample = self.samples[index]
        if self.use_cache:
            filepath = sample["audio"] + ".pt"
            if os.path.exists(filepath):
                obj = torch.load(filepath)
            else:
                obj = self.get_audio_text_duration_f0(sample)
                torch.save(obj,filepath)
            return obj
        else:
            return self.get_audio_text_duration_f0(sample)

    def __len__(self):
        return len(self.samples)

    def clear_cache(self):
        """clear all the cache"""
        for sample in self.samples:
            filepath = sample["audio"] + ".pt"
            if os.path.exists(filepath):
                os.remove(filepath)

    def get_audio_text_duration_f0(self, sample):
        tokens, phoneme = self._get_text(sample["text"])
        wav = self.processor.load_wav(sample["audio"], sr=self.sample_rate)
        wav_t = torch.FloatTensor(wav)
        wav_t = wav_t.unsqueeze(0)

        spec = None
        if self.add_spec:
            spec = self.processor.spectrogram(wav)
            spec = torch.FloatTensor(spec)
        mel = None
        if self.add_mel:
            mel = self.processor.melspectrogram(wav)
            mel = torch.FloatTensor(mel)

        #TODO:change code here
        duration = torch.zeros_like(tokens)
        f0 = torch.zeros_like(tokens)
        return {
            "raw_text": sample["text"],
            "phoneme": phoneme,
            "tokens": tokens,
            "token_len": len(tokens),
            "wav": wav_t,
            "spec": spec,
            "mel": mel,
            "audio_file": sample["audio"],
            "speaker": sample["speaker"],
            "duration": duration,
            "f0": f0
        }

    # def _get_audio(self, filename):
    #     wav, sample_rate = torchaudio.load(filename)
    #
    #     if sample_rate != self.sample_rate:
    #         raise ValueError( "{} SR doesn't match target {} SR".format( sample_rate, self.sample_rate) )
    #     return wav

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

    def collate_fn(self, batch):
        """Zero-pads model inputs, audios and targets and pad a batch and sort by wav decreasing
        """
        B = len(batch)
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["wav"].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([x["tokens"].size(0) for x in batch])
        token_lens = torch.LongTensor([x["token_len"] for x in batch])
        wav_lens = torch.LongTensor([x["wav"].size(1) for x in batch])
        wav_lens_max = torch.max(wav_lens)
        spec_feat_len = batch[0]["spec"].size(0) if self.add_spec else 10
        spec_lens = torch.LongTensor([x["spec"].size(1) for x in batch]) if self.add_spec else torch.LongTensor([10 for x in batch])
        spec_lens_max = torch.max(spec_lens)
        mel_feat_len = batch[0]["mel"].size(0) if self.add_mel else 10
        mel_lens = torch.LongTensor([x["mel"].size(1) for x in batch]) if self.add_mel else torch.LongTensor([10 for x in batch])
        mel_lens_max = torch.max(mel_lens)

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        spec_padded = torch.FloatTensor(B, spec_feat_len, spec_lens_max)
        mel_padded = torch.FloatTensor(B, mel_feat_len, mel_lens_max)

        token_padded = token_padded.zero_()
        wav_padded = wav_padded.zero_()
        spec_padded = spec_padded.zero_()
        mel_padded = mel_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            item = batch[ids_sorted_decreasing[i]]

            tokens = item["tokens"]
            token_padded[i, : tokens.size(0)] = torch.LongTensor(tokens)
            wav = item["wav"]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)
            if self.add_spec:
                spec = item["spec"]
                spec_padded[i, :, :spec.size(1)] = torch.FloatTensor(spec)
            if self.add_mel:
                mel = item["mel"]
                mel_padded[i, :, :mel.size(1)] = torch.FloatTensor(mel)

        return {
            "tokens": token_padded, # [B, T]
            "token_lens": token_lens, # [B]
            "waveform": wav_padded, # [B, 1, T_wav]
            "waveform_lens": wav_lens,# [B]
            "spec": spec_padded, #[B, C, T_spec]
            "spec_lens": spec_lens, # [B]
            "mel": mel_padded, # [B, C, T_mel]
            "mel_lens": mel_lens, # [B]
            "speakers": [x["speaker"] for x in batch],# [B]
            "audio_files": [x["audio_file"] for x in batch],# [B]
            "raw_texts": [x["raw_text"] for x in batch] # [B]
        }
