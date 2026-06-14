"""Data augmentation module for audio using librosa."""
import random
import numpy as np
import librosa
from src.config import AUGMENTATION_CONFIG


def time_stretch(y, sr, factor):
    """Stretch audio in time by given factor using librosa.

    Args:
        y: Audio time series (numpy array)
        sr: Sample rate
        factor: Stretch factor (>1 slower, <1 faster)

    Returns:
        Stretched audio array
    """
    return librosa.effects.time_stretch(y, rate=factor)


def pitch_shift(y, sr, n_steps):
    """Shift pitch by n_steps semitones using librosa.

    Args:
        y: Audio time series (numpy array)
        sr: Sample rate
        n_steps: Number of semitones to shift (positive = up, negative = down)

    Returns:
        Pitch-shifted audio array
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def add_noise(y, snr_db):
    """Add white noise at specified signal-to-noise ratio.

    Args:
        y: Audio time series (numpy array)
        snr_db: Signal-to-noise ratio in decibels

    Returns:
        Audio with added noise
    """
    # Calculate signal power
    signal_power = np.mean(y ** 2)

    # Calculate noise power from SNR
    # snr_db = 10 * log10(signal_power / noise_power)
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate white noise
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)

    return y + noise


def volume_scale(y, scale):
    """Scale audio amplitude by given factor.

    Args:
        y: Audio time series (numpy array)
        scale: Scaling factor for amplitude

    Returns:
        Volume-scaled audio array
    """
    return y * scale


def time_shift(y, shift_max=0.2):
    """Randomly roll audio within ±shift_max of length.

    Args:
        y: Audio time series (numpy array)
        shift_max: Maximum shift as fraction of audio length

    Returns:
        Time-shifted audio array
    """
    shift_frac = random.uniform(-shift_max, shift_max)
    shift_samples = int(len(y) * shift_frac)
    return np.roll(y, shift_samples)


def augment_sample(y, sr, n_augmentations=None):
    """Apply random augmentations to an audio sample.

    Each augmentation is applied independently to a copy of the original,
    generating n_augmentations different augmented versions.

    Args:
        y: Audio time series (numpy array)
        sr: Sample rate
        n_augmentations: Number of augmented versions to generate.
                          Defaults to config value.

    Returns:
        List of augmented audio arrays (does not include original)
    """
    if n_augmentations is None:
        n_augmentations = AUGMENTATION_CONFIG["max_aug_per_sample"]

    augmented = []

    # Select random augmentation types for each output
    for _ in range(n_augmentations):
        aug_type = random.choice([
            "time_stretch",
            "pitch_shift",
            "noise",
            "volume",
            "time_shift"
        ])

        y_aug = y.copy()

        if aug_type == "time_stretch":
            factor = random.choice(AUGMENTATION_CONFIG["time_stretch_factors"])
            y_aug = time_stretch(y_aug, sr, factor)

        elif aug_type == "pitch_shift":
            n_steps = random.choice(AUGMENTATION_CONFIG["pitch_shift_steps"])
            y_aug = pitch_shift(y_aug, sr, n_steps)

        elif aug_type == "noise":
            snr_db = random.choice(AUGMENTATION_CONFIG["noise_snrs_db"])
            y_aug = add_noise(y_aug, snr_db)

        elif aug_type == "volume":
            scale = random.choice(AUGMENTATION_CONFIG["volume_scales"])
            y_aug = volume_scale(y_aug, scale)

        elif aug_type == "time_shift":
            y_aug = time_shift(y_aug)

        augmented.append(y_aug)

    return augmented