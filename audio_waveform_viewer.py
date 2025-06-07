import os
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from datetime import datetime

# Configuration
PROJECT_ROOT = r"C:\Users\jayac\PycharmProjects\PythonProject"
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "Record & predict", "Recorded_audio")
LAST_RECORDING_FILE = os.path.join(PROJECT_ROOT, "Record & predict", "last_recording.txt")


def get_last_recording():
    """Get path to last recording if exists"""
    if os.path.exists(LAST_RECORDING_FILE):
        with open(LAST_RECORDING_FILE, 'r') as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return None


def show_waveform(audio_path):
    """Display waveform of audio file"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert to numpy array
    audio_data = waveform.numpy()[0]

    # Create time axis
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, num=len(audio_data))

    # Plot waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio_data, linewidth=0.5)
    plt.title(f"Audio Waveform: {os.path.basename(audio_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def list_recordings():
    """List all available recordings"""
    if not os.path.exists(RECORDINGS_DIR):
        print("No recordings directory found!")
        return []

    files = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.wav')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(RECORDINGS_DIR, x)))

    print("\nAvailable recordings:")
    for i, file in enumerate(files, 1):
        filepath = os.path.join(RECORDINGS_DIR, file)
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        print(f"{i}. {file} (Recorded: {mtime:%Y-%m-%d %H:%M:%S})")

    return files


def main():
    print("=== Audio Waveform Viewer ===")

    # Check for last recording
    last_recording = get_last_recording()
    if last_recording:
        print(f"\nLast recording: {os.path.basename(last_recording)}")

    # List all recordings
    files = list_recordings()

    if not files:
        return

    try:
        selection = input("\nEnter recording number to view (or 'last'): ").strip()

        if selection.lower() == 'last' and last_recording:
            show_waveform(last_recording)
        elif selection.isdigit():
            index = int(selection) - 1
            if 0 <= index < len(files):
                filepath = os.path.join(RECORDINGS_DIR, files[index])
                show_waveform(filepath)
            else:
                print("Invalid selection!")
        else:
            print("Please enter a valid number or 'last'")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()