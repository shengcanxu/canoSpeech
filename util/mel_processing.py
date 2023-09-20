import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf

"""
do the similar audio process with util.audio_processor.py, but do it in GPU. all the functions here should be call in GPU
so that it's suitable to call in the format_batch_on_device callback in Trainer
"""

mel_basis = {}
hann_window = {}

def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output

def wav_to_spec(wav, n_fft, hop_size, win_size, center=False):
    """
    Args Shapes:
        - wav : :math:`[B, 1, T]`
    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    wav = wav.squeeze(1)
    if torch.min(wav) < -1.0:
        print("min value is ", torch.min(wav))
    if torch.max(wav) > 1.0:
        print("max value is ", torch.max(wav))

    global hann_window
    dtype_device = str(wav.dtype) + "_" + str(wav.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=wav.dtype, device=wav.device
        )

    wav = torch.nn.functional.pad(
        wav.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    wav = wav.squeeze(1)

    spec = torch.stft(
        input=wav,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`
    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel


def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`
    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        input=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]
    Return Shapes:
        - x: :math:`[1, T]`
    """
    # x, sr = torchaudio.load(file_path)
    # assert (x > 1).sum() + (x < -1).sum() == 0
    # return x, sr
    x, sr = sf.read(file_path)
    x = torch.FloatTensor(x)
    x = x.unsqueeze(0)
    return x, sr