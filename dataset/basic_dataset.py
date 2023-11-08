import os
import random
import numpy as np
from collections import Counter
import pickle

from dataset.dataset_constant import VCTK_speaker_id_mapping
from text import cleaned_text_to_tokens, _clean_text, _intersperse
import torch
from torch.utils.data import Dataset
import librosa
from util.mel_processing import wav_to_mel, wav_to_spec, spec_to_mel, load_audio


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
    ), " [!] You do not have enough samples for the evaluation set. You can work` around this setting the 'eval_split_size' parameter to a minimum of {}".format(1 / len(items) )

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
    """get meta from filelist, filelist is generated from gen_filelist.py"""
    metas = []
    with open(filelist, encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("|")
            metas.append({
                "text": items[3],
                "language": items[2],
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
        self.dataset_name = config.dataset_config.dataset_name
        self.use_cache = getattr(config.dataset_config, "use_cache", False)
        self.use_speaker_ids = config.model.use_speaker_ids
        self.melspec_use_GPU = getattr(config.dataset_config, "melspec_use_GPU", False)
        self.add_preprocess_data = getattr(config.dataset_config, "add_preprocess_data", True)

        self.hop_length = config.audio.hop_length
        self.win_length = config.audio.win_length
        self.sample_rate = config.audio.sample_rate
        self.fft_size = config.audio.fft_size
        self.num_mels = config.audio.num_mels
        self.mel_fmin = config.audio.mel_fmin
        self.mel_fmax = config.audio.mel_fmax
        self.max_audio_length = getattr(config.audio, "max_audio_length", 10.0)
        self.min_audio_length = getattr(config.audio, "min_audio_length", 1.0)

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
        for sample in self.samples:
            audio_len = librosa.get_duration(path=sample["audio"])
            # audio_len should less than config.audio.max_audio_length. it controls the max dimention of mel
            # max_mel_len = max_audio_length * sample_rate / hop_length
            if self.min_text_len <= len(sample["text"]) <= self.max_text_len\
                    and self.min_audio_length <= audio_len <= self.max_audio_length:
                samples_new.append(sample)
        self.samples = samples_new

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
        wav, sr = load_audio(sample["audio"])

        spec, mel = None, None
        if not self.melspec_use_GPU:
            spec = wav_to_spec(wav, self.fft_size, self.hop_length, self.win_length)
            mel = spec_to_mel(spec, self.fft_size, self.num_mels, self.sample_rate, self.mel_fmin, self.mel_fmax)
            spec = torch.FloatTensor(spec)
            mel = torch.FloatTensor(mel)

        pitch, duration, speaker = None, None, None
        if self.add_preprocess_data:
            path = sample["audio"] + ".pkl"
            if os.path.exists(path):
                fp = open(path, "rb")
                pickleObj = pickle.load(fp)
                pitch = torch.FloatTensor(pickleObj["pitch"])
                speaker = torch.FloatTensor(pickleObj["speaker"])
            else:
                raise Exception("path doesn't exists! should run preprocess")

        return {
            "raw_text": sample["text"], # str
            "phoneme": phoneme, # str
            "tokens": tokens,
            "token_len": len(tokens),
            "wav": wav,
            "spec": spec.squeeze(),
            "mel": mel.squeeze(),
            "filename": sample["audio"],
            "speaker_id": self._get_speaker_id(sample["speaker"], self.dataset_name),
            "speaker_embed": speaker,
            "pitch": pitch,
            "duration": duration
        }

    def _get_speaker_id(self, speaker_name, dataset_name:str):
        if dataset_name == "vctk":
            return VCTK_speaker_id_mapping.get(speaker_name, 1)
        else:
            return None

    def _get_text(self, text):
        """format text and add blank"""
        if self.cleaned_text:
            cleaned_text = text
            tokens = cleaned_text_to_tokens(cleaned_text)
        else:
            cleaned_text = _clean_text(text, self.text_cleaners)
            tokens = cleaned_text_to_tokens(cleaned_text)
        if self.add_blank:
            tokens = _intersperse(tokens, 0)
        tokens = torch.LongTensor(tokens)
        return tokens, cleaned_text

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
        token_padded = torch.LongTensor(B, max_text_len)
        token_padded = token_padded.zero_()

        speaker_ids = torch.LongTensor([x["speaker_id"] for x in batch]) if self.use_speaker_ids else None
        wav_lens = torch.LongTensor([x["wav"].size(1) for x in batch])
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        wav_padded = wav_padded.zero_()

        spec_lens, mel_lens = [], []
        spec_padded, mel_padded = None, None
        if not self.melspec_use_GPU:  # if mel spec generated using GPU, it will be generate in format_batch_on_device callback
            spec_feat_len = batch[0]["spec"].size(0)
            spec_lens = torch.LongTensor([x["spec"].size(1) for x in batch])
            spec_lens_max = torch.max(spec_lens)
            spec_padded = torch.FloatTensor(B, spec_feat_len, spec_lens_max)
            spec_padded = spec_padded.zero_()

            mel_feat_len = batch[0]["mel"].size(0)
            mel_lens = torch.LongTensor([x["mel"].size(1) for x in batch])
            mel_lens_max = torch.max(mel_lens)
            mel_padded = torch.FloatTensor(B, mel_feat_len, mel_lens_max)
            mel_padded = mel_padded.zero_()

        pitch_padded, duration_padded, speaker_embed_padded = None, None, None
        should_pad_duration = True
        for item in batch:
            if item["duration"] is None:
                should_pad_duration = False

        if self.add_preprocess_data:
            pitch_lens = torch.LongTensor([x["pitch"].size(0) for x in batch])
            pitch_lens_max = torch.max(pitch_lens)
            pitch_padded = torch.FloatTensor(B, pitch_lens_max)
            pitch_padded = pitch_padded.zero_()

            if should_pad_duration:
                duration_lens = torch.LongTensor([x["duration"].size(0) for x in batch])
                duration_lens_max = torch.max(duration_lens)
                duration_padded = torch.IntTensor(B, duration_lens_max)
                duration_padded = duration_padded.zero_()

            speaker_embed_lens = torch.LongTensor([x["speaker_embed"].size(0) for x in batch])
            speaker_embed_lens_max = torch.max(speaker_embed_lens)
            speaker_embed_padded = torch.FloatTensor(B, speaker_embed_lens_max)
            speaker_embed_padded = speaker_embed_padded.zero_()

        filenames, raw_texts = [], []
        for i in range(len(ids_sorted_decreasing)):
            item = batch[ids_sorted_decreasing[i]]
            filenames.append(item["filename"])
            raw_texts.append(item["raw_text"])

            tokens = item["tokens"]
            token_padded[i, : tokens.size(0)] = torch.LongTensor(tokens)
            wav = item["wav"]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)
            if not self.melspec_use_GPU:
                spec = item["spec"]
                spec_padded[i, :, :spec.size(1)] = torch.FloatTensor(spec)
                mel = item["mel"]
                mel_padded[i, :, :mel.size(1)] = torch.FloatTensor(mel)

            if self.add_preprocess_data:
                pitch = item["pitch"]
                pitch_padded[i, :pitch.size(0)] = torch.FloatTensor(pitch)
                if should_pad_duration:
                    duration = item["duration"]
                    duration_padded[i, :duration.size(0)] = duration
                speaker_embed = item["speaker_embed"]
                speaker_embed_padded[i, :speaker_embed.size(0)] = torch.FloatTensor(speaker_embed)

        return {
            "tokens": token_padded,  # [B, T]
            "token_lens": token_lens,  # [B]
            "waveform": wav_padded,  # [B, 1, T_wav]
            "waveform_lens": wav_lens,  # [B]
            "waveform_rel_lens": wav_rel_lens,  #[B], wave len in percentage
            "spec": spec_padded,  #[B, C, T_spec]
            "spec_lens": spec_lens,  # [B]
            "mel": mel_padded,  # [B, C, T_mel]
            "mel_lens": mel_lens,  # [B]
            "speaker_ids": speaker_ids,  # [B]
            "speaker_embed": speaker_embed_padded,  # [B, T_speaker]
            "pitch": pitch_padded,  # [B, T_pitch]
            "duration": duration_padded,  #[B, T_duration]
            "filenames": filenames,  # [B]
            "raw_texts": raw_texts  # [B]
        }