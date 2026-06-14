#!/usr/bin/env python3
"""
Voice Data Collector v2
Interactive CLI tool for recording voice samples with progress feedback.
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CLASSES, SAMPLE_RATE, DATASET_PATH

# ── Collector Settings ──
SAMPLES_PER_CLASS = 100
DURATION = 1.5       # seconds per sample
COUNTDOWN = 1        # seconds before recording starts
VAD_THRESHOLD = 0.005


def normalize_audio(audio: np.ndarray, target_max: int = 32767) -> np.ndarray:
    """Normalize audio to int16 range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio * (target_max / max_val)
    return audio.astype(np.int16)


def get_device() -> int:
    """Query and return the default input device index."""
    devices = sd.query_devices(kind='input')
    return devices.get('default_input_device', 0)


def record_sample(class_name: str, sample_idx: int) -> bool:
    """
    Record a single voice sample with countdown and VAD check.

    Returns:
        True if sample was recorded and saved successfully.
    """
    device = get_device()
    device_info = sd.query_devices(device, 'input')
    channels = device_info.get('max_input_channels', 1)

    class_dir = os.path.join(DATASET_PATH, class_name)
    os.makedirs(class_dir, exist_ok=True)

    file_path = os.path.join(class_dir, f"{class_name}_{sample_idx:04d}.wav")

    # Countdown
    print(f"\n  Recording '{class_name}' sample {sample_idx}/{SAMPLES_PER_CLASS} in:", end=" ")
    for sec in range(COUNTDOWN, 0, -1):
        print(f"{sec}", end=" ", flush=True)
        time.sleep(1)
    print("GO!")

    # Record
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype='float32',
        device=device
    )
    sd.wait()

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # VAD check
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < VAD_THRESHOLD:
        print("  [Low audio detected - resampling...]")
        return False

    # Normalize and save
    audio_int16 = normalize_audio(audio)
    wavfile.write(file_path, SAMPLE_RATE, audio_int16)
    return True


def record_all_classes():
    """Record SAMPLES_PER_CLASS samples for each class."""
    total = len(CLASSES) * SAMPLES_PER_CLASS
    print(f"\n{'='*50}")
    print(f"  Recording {SAMPLES_PER_CLASS} samples per class")
    print(f"  Total samples: {total}")
    print(f"{'='*50}\n")

    with tqdm(total=total, desc="Overall Progress", unit="sample") as pbar:
        for class_name in CLASSES:
            pbar.set_description(f"Recording {class_name}")
            class_dir = os.path.join(DATASET_PATH, class_name)
            existing = len([f for f in os.listdir(class_dir)
                           if f.startswith(class_name) and f.endswith('.wav')]) \
                       if os.path.exists(class_dir) else 0

            for i in range(SAMPLES_PER_CLASS):
                sample_idx = existing + i + 1
                success = False
                while not success:
                    success = record_sample(class_name, sample_idx)
                pbar.update(1)

    print("\n  Recording complete!")


def record_single_class():
    """Interactively record samples for a single selected class."""
    print("\n  Available classes:")
    for i, cls in enumerate(CLASSES, 1):
        print(f"    {i}. {cls}")

    try:
        choice = int(input("\n  Select class number: ")) - 1
        if choice < 0 or choice >= len(CLASSES):
            print("  Invalid selection.")
            return
    except ValueError:
        print("  Invalid input.")
        return

    class_name = CLASSES[choice]
    class_dir = os.path.join(DATASET_PATH, class_name)
    os.makedirs(class_dir, exist_ok=True)
    existing = len([f for f in os.listdir(class_dir)
                   if f.startswith(class_name) and f.endswith('.wav')]) \
               if os.path.exists(class_dir) else 0

    print(f"\n  Recording for class '{class_name}'")
    print(f"  Existing samples: {existing}")
    print(f"  Target: {SAMPLES_PER_CLASS} total\n")

    for i in range(SAMPLES_PER_CLASS):
        sample_idx = existing + i + 1
        success = False
        attempts = 0
        while not success and attempts < 3:
            success = record_sample(class_name, sample_idx)
            attempts += 1
        if not success:
            print(f"  Skipping sample {sample_idx} after 3 failed attempts.")

    print("\n  Done!")


def custom_record():
    """Record a custom number of samples for all classes."""
    global SAMPLES_PER_CLASS
    try:
        count = int(input(f"\n  Samples per class (default {SAMPLES_PER_CLASS}): ") or SAMPLES_PER_CLASS)
    except ValueError:
        print("  Invalid count.")
        return

    SAMPLES_PER_CLASS = count

    total = len(CLASSES) * SAMPLES_PER_CLASS
    print(f"\n  Recording {SAMPLES_PER_CLASS} samples per class")
    print(f"  Total: {total} samples\n")

    with tqdm(total=total, desc="Overall Progress", unit="sample") as pbar:
        for class_name in CLASSES:
            pbar.set_description(f"Recording {class_name}")
            class_dir = os.path.join(DATASET_PATH, class_name)
            existing = len([f for f in os.listdir(class_dir)
                           if f.startswith(class_name) and f.endswith('.wav')]) \
                       if os.path.exists(class_dir) else 0

            for i in range(SAMPLES_PER_CLASS):
                sample_idx = existing + i + 1
                success = False
                while not success:
                    success = record_sample(class_name, sample_idx)
                pbar.update(1)

    print("\n  Recording complete!")


def test_microphone():
    """Test microphone with a 3-second recording."""
    print("\n  Testing microphone...")
    print("  Speak now for 3 seconds.\n")

    device = get_device()
    device_info = sd.query_devices(device, 'input')
    channels = device_info.get('max_input_channels', 1)

    audio = sd.rec(
        int(3 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype='float32',
        device=device
    )

    with tqdm(total=3, desc="Recording", unit="s") as pbar:
        start = time.time()
        while sd.get_stream().active:
            time.sleep(0.1)
            elapsed = int(time.time() - start)
            if elapsed < 3:
                pbar.update(elapsed - pbar.n)
            else:
                break

    sd.wait()

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rms = np.sqrt(np.mean(audio ** 2))
    max_amp = np.abs(audio).max()

    print(f"\n  RMS Level: {rms:.4f}")
    print(f"  Max Amplitude: {max_amp:.4f}")
    print(f"  Status: {'OK' if rms > VAD_THRESHOLD else 'Too quiet - check mic'}")

    temp_path = os.path.join(DATASET_PATH, "mic_test.wav")
    os.makedirs(DATASET_PATH, exist_ok=True)
    audio_int16 = normalize_audio(audio)
    wavfile.write(temp_path, SAMPLE_RATE, audio_int16)
    print(f"  Test recording saved to: {temp_path}")


def print_summary():
    """Print dataset summary."""
    print("\n  Dataset Summary")
    print(f"  {'─'*40}")
    print(f"  Path: {DATASET_PATH}")
    print(f"  Classes: {len(CLASSES)}")
    print(f"  Class list: {', '.join(CLASSES)}")
    print(f"  {'─'*40}")

    total_files = 0
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_PATH, class_name)
        count = len([f for f in os.listdir(class_dir)
                    if f.startswith(class_name) and f.endswith('.wav')]) \
                if os.path.exists(class_dir) else 0
        total_files += count
        bar = '█' * (count // 5) + '░' * (20 - count // 5)
        print(f"  {class_name:10s}: {count:4d} [{bar}]")

    print(f"  {'─'*40}")
    print(f"  Total: {total_files} samples")
    print()


def main():
    """Interactive main menu."""
    menu = f"""
╔════════════════════════════════════════╗
║     Voice Data Collector v2            ║
║     {'Classes: ' + str(len(CLASSES)):26s} ║
║     {'Samples/Class: ' + str(SAMPLES_PER_CLASS):24s} ║
║     {'Sample Rate: ' + str(SAMPLE_RATE):24s} ║
║     {'Duration: ' + str(DURATION) + 's':27s} ║
╚════════════════════════════════════════╝

  1. Record all classes
  2. Record single class
  3. Custom count
  4. Test microphone
  5. Summary
  6. Quit
"""

    while True:
        print(menu)
        choice = input("  Select option: ").strip()

        if choice == '1':
            record_all_classes()
        elif choice == '2':
            record_single_class()
        elif choice == '3':
            custom_record()
        elif choice == '4':
            test_microphone()
        elif choice == '5':
            print_summary()
        elif choice == '6':
            print("\n  Goodbye!\n")
            break
        else:
            print("\n  Invalid option.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!\n")
        sys.exit(0)