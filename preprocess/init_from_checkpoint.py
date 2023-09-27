import fsspec
import torch

if __name__ == "__main__":
    path = "D:\\project\\canoSpeech\\preprocess\\reference\\vits_pretrained_vctk.pth"
    with fsspec.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    a = 3