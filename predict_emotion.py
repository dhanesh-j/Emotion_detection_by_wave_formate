import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import sys

# Force CPU for stability
device = torch.device("cpu")
print("⚠️ Using CPU for inference")

print("\nLoading model...")
model_dir = "final_emotion_model"
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(device)
print("✅ Model loaded successfully")

if len(sys.argv) < 2:
    print("❌ Please provide a path to the audio file.")
    sys.exit(1)

file_path = sys.argv[1]

print(f"Loading audio file: {file_path}")
waveform, sample_rate = torchaudio.load(file_path)
print(f"Original waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
print(f"Mono waveform shape: {waveform.shape}")

if sample_rate != 16000:
    print("Resampling audio to 16kHz...")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    print(f"Resampled waveform shape: {waveform.shape}")

inputs = processor(
    waveform.numpy()[0],
    sampling_rate=16000,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=16000 * 4  # 4 seconds
)

print(f"Processed input keys: {inputs.keys()}")

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("Running model inference...")
    logits = model(**inputs).logits
    print(f"Logits: {logits}")
    predicted_id = torch.argmax(logits, dim=-1).item()

id2label = model.config.id2label
predicted_emotion = id2label[predicted_id]

print(f"\n🎯 Predicted Emotion: {predicted_emotion}")


