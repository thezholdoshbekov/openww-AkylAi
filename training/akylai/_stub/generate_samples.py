"""No-op stub so openwakeword/train.py can `from generate_samples import generate_samples`
without cloning piper-sample-generator. We never pass --generate_clips, so this is only
imported, never called."""


def generate_samples(*args, **kwargs):
    raise RuntimeError(
        "Synthetic generation is disabled for the akylai pipeline. "
        "Do not pass --generate_clips; clips are supplied by prepare_data.py."
    )
