import os
import time
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from datetime import datetime
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Configuration - using absolute paths
PROJECT_ROOT = r"C:\Users\jayac\PycharmProjects\PythonProject"
CONFIG = {
    "recording_dir": os.path.join(PROJECT_ROOT, "Record & predict", "Recorded_audio"),
    "model_dir": os.path.join(PROJECT_ROOT, "final_emotion_model"),
    "sample_rate": 16000,
    "duration": 4,  # seconds
    "countdown": 3,  # 3-second countdown
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "file_prefix": "03-01",
}


class AudioRecorder:
    @staticmethod
    def ensure_directory_exists(directory):
        """Create directory if it doesn't exist"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    @classmethod
    def countdown(cls, seconds):
        """Visual countdown before recording"""
        for i in range(seconds, 0, -1):
            print(f"⏳ Recording starts in {i}...", end='\r')
            time.sleep(1)
        print("🎤 Start speaking now!" + " " * 20)  # Clear line

    @classmethod
    def record_audio(cls):
        """Record audio from microphone with countdown"""
        print(f"\nPreparing to record {CONFIG['duration']} seconds of audio...")
        cls.countdown(CONFIG['countdown'])

        audio = sd.rec(
            int(CONFIG['duration'] * CONFIG['sample_rate']),
            samplerate=CONFIG['sample_rate'],
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("✅ Recording complete")
        return audio.flatten()

    @classmethod
    def save_recording(cls, audio):
        """Save audio in Recorded_audio directory"""
        cls.ensure_directory_exists(CONFIG['recording_dir'])

        timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
        filename = f"{CONFIG['file_prefix']}-{timestamp}.wav"
        filepath = os.path.join(CONFIG['recording_dir'], filename)

        waveform = torch.from_numpy(audio).unsqueeze(0)
        torchaudio.save(filepath, waveform, CONFIG['sample_rate'])
        print(f"💾 Saved recording to: {filepath}")
        return filepath


class EmotionPredictor:
    def __init__(self):
        print("\nInitializing emotion detection model...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG['model_dir'])
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(CONFIG['model_dir']).to(CONFIG['device'])
        print(f"✅ Model loaded on {CONFIG['device']}")

    def predict(self, audio_path):
        """Predict emotion from audio file"""
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != CONFIG['sample_rate']:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=CONFIG['sample_rate'])
            waveform = resampler(waveform)

        inputs = self.processor(
            waveform.numpy()[0],
            sampling_rate=CONFIG['sample_rate'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=CONFIG['sample_rate'] * CONFIG['duration']
        )

        with torch.no_grad():
            inputs = {k: v.to(CONFIG['device']) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            predicted_id = torch.argmax(logits, dim=-1).item()

        return self.model.config.id2label[predicted_id]


def main():
    print("\n=== Voice Emotion Detection ===")
    print(f"Recordings will be saved to: {CONFIG['recording_dir']}")

    try:
        # 1. Record with countdown
        audio_data = AudioRecorder.record_audio()

        # 2. Save recording
        audio_path = AudioRecorder.save_recording(audio_data)

        # 3. Predict emotion
        predictor = EmotionPredictor()
        emotion = predictor.predict(audio_path)
        print(f"\n🎯 Predicted Emotion: {emotion}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    finally:
        print("\n=== Session Ended ===")


if __name__ == "__main__":
    main()