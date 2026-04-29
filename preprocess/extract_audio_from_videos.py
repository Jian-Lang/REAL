import subprocess
from pathlib import Path


for dataset in ["FakeTT", "FVC", "FakeSV"]:
    video_dir = Path(f"data/{dataset}/videos")
    audio_dir = Path(f"data/{dataset}/audios")
    audio_dir.mkdir(parents=True, exist_ok=True)

    for video_path in video_dir.glob("*.mp4"):
        audio_path = audio_dir / f"{video_path.stem}.wav"
        if audio_path.exists():
            continue

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path),
            ],
            check=True,
        )
