from huggingface_hub import snapshot_download
import os
import huggingface_hub
# huggingface_hub.login("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")

os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"

snapshot_download(
  repo_id="pyannote/speaker-diarization",
  revision="2.1",
  local_dir="D:/models/pyannote/speaker-diarization",
  token="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",
  # proxies={"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"},
  local_dir_use_symlinks=False,
  max_workers=8
)
