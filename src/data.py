"""Loading speechocean762 and inspecting its phone inventory.

speechocean762 supplies, per utterance:
  - text: the target sentence (already uppercase, no punctuation)
  - words: list of {text, phones, phones-accuracy, accuracy, stress, total,
    mispronunciations}, giving the canonical ARPAbet-with-stress-digit phone
    sequence per word and a 0-2 human accuracy score per phone.
  - accuracy / completeness / fluency / prosodic / total: utterance-level
    0-10 scores.
  - speaker, gender, age: speaker metadata (speaker id is what we split on
    for a speaker-disjoint test set -- never split by utterance).
  - audio: decoded waveform (see _register_ffmpeg_dll_dir for why this needs
    a Windows-specific fix to work with torchcodec).
"""
import glob
import os
import sys

SEED = 42

DATA_FILES = {
    "train": "datasets/speechocean762/data/train-00000-of-00001.parquet",
    "test": "datasets/speechocean762/data/test-00000-of-00001.parquet",
}


def _register_ffmpeg_dll_dir():
    """On Windows, Python 3.8+ no longer searches PATH for a loaded DLL's
    dependencies, so torchcodec can fail to find FFmpeg's shared libraries
    even when ffmpeg is on PATH. Explicitly register the directory that
    holds avcodec-*.dll (the FFmpeg "shared" build) via os.add_dll_directory.
    """
    if sys.platform != "win32":
        return

    candidate_dirs = list(os.environ.get("PATH", "").split(os.pathsep))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate_dirs += glob.glob(
            os.path.join(
                local_app_data,
                "Microsoft", "WinGet", "Packages",
                "Gyan.FFmpeg.Shared_*", "ffmpeg-*", "bin",
            )
        )

    seen = set()
    for path_dir in candidate_dirs:
        if path_dir and path_dir not in seen and glob.glob(os.path.join(path_dir, "avcodec-*.dll")):
            seen.add(path_dir)
            try:
                os.add_dll_directory(path_dir)
            except (OSError, FileNotFoundError):
                pass


_register_ffmpeg_dll_dir()

from datasets import load_dataset, concatenate_datasets  # noqa: E402


def load_speechocean762(decode_audio: bool = True):
    ds = load_dataset("parquet", data_files=DATA_FILES)
    if not decode_audio:
        ds = ds.cast_column("audio", ds["train"].features["audio"].__class__(decode=False))
    return ds


def all_rows(ds):
    """train + test concatenated, for inventory scans that don't care about the split."""
    return concatenate_datasets([ds["train"], ds["test"]])


def get_phone_inventory(ds):
    """Distinct canonical phone symbols (ARPAbet + stress digit, e.g. 'IY0')
    that appear anywhere in the dataset's per-word 'phones' lists."""
    inventory = set()
    for split in ("train", "test"):
        for words in ds[split]["words"]:
            for word in words:
                inventory.update(word["phones"])
    return sorted(inventory)


def print_example_summary(example, index=0):
    print(f"=== Example #{index} ===")
    print("Top-level field names:", list(example.keys()))
    print(f"text: {example['text']}")
    print(f"speaker: {example['speaker']}  gender: {example['gender']}  age: {example['age']}")
    print(f"utterance scores: accuracy={example['accuracy']} completeness={example['completeness']} "
          f"fluency={example['fluency']} prosodic={example['prosodic']} total={example['total']}")

    audio = example["audio"]
    try:
        arr = audio["array"]
        sr = audio["sampling_rate"]
        print(f"audio: sampling_rate={sr} Hz, shape={arr.shape}, duration={len(arr) / sr:.2f}s")
    except (TypeError, KeyError):
        print(f"audio (not decoded): {type(audio)}")

    print("words:")
    for w in example["words"]:
        print(f"  {w['text']!r}: phones={w['phones']} phones-accuracy={w['phones-accuracy']} "
              f"word_accuracy={w['accuracy']} stress={w['stress']} mispronunciations={w['mispronunciations']}")


if __name__ == "__main__":
    ds = load_speechocean762(decode_audio=True)
    print(ds)
    print()
    print_example_summary(ds["train"][0], index=0)

    print()
    inventory = get_phone_inventory(ds)
    print(f"speechocean762 phone inventory ({len(inventory)} distinct symbols):")
    print(inventory)
