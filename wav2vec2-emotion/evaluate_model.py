import torch
import torchaudio
import torch_directml
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer
)
from datasets import Dataset
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import warnings
import os

# Setup
warnings.filterwarnings("ignore")

# Initialize device
try:
    device = torch_directml.device()
    print("AMD GPU detected - Using DirectML backend")
except Exception as error:
    device = torch.device("cpu")
    print(f"AMD GPU not available - Falling back to CPU. Error: {str(error)}")

# Correct paths
base_path = r"C:\Users\jayac\PycharmProjects\PythonProject"
model_path = os.path.join(base_path, "final_emotion_model")
csv_path = os.path.join(base_path, "ravdess_data.csv")  # CSV is in base directory
audio_base_path = base_path  # Assuming audio files are also in base directory

print(f"\nLoading trained model from {model_path}...")

# Verify model directory
if not os.path.exists(model_path):
    raise ValueError(f"Model directory not found at: {model_path}")

print("Found these files in model directory:", os.listdir(model_path))

try:
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    print("Model loaded successfully")

    # Get class mappings
    id2label = model.config.id2label
    label2id = model.config.label2id
    class_names = list(label2id.keys())
    print(f"Class labels: {class_names}")

except Exception as error:
    raise ValueError(f"Error loading model: {str(error)}")

# Load test data
print("\nLoading test data...")
try:
    test_df = pd.read_csv(csv_path)
    test_df['label'] = test_df['emotion'].map(label2id)
    test_dataset = Dataset.from_pandas(test_df)
    print(f"Loaded {len(test_df)} test samples from {csv_path}")
    print("Sample paths from CSV:", test_df['path'].head())  # Debug: show sample paths
except Exception as error:
    raise ValueError(f"Test data loading error: {str(error)}\n"
                     f"Expected CSV at: {csv_path}\n"
                     f"Directory contents: {os.listdir(base_path)}")


# Preprocessing function with absolute path handling
def preprocess_function(batch):
    audio_arrays = []
    valid_indices = []

    for idx, path in enumerate(batch["path"]):
        try:
            # Handle paths - assuming paths in CSV are relative to base_path
            full_path = os.path.join(audio_base_path, path)
            print(f"Loading audio from: {full_path}")  # Debug print

            # Verify file exists
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Audio file not found: {full_path}")

            waveform, sample_rate = torchaudio.load(full_path)
            waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            audio_arrays.append(waveform.numpy()[0])
            valid_indices.append(idx)
        except Exception as error:
            print(f"Error processing {path}: {str(error)}")
            continue

    if not audio_arrays:
        return None

    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        padding="max_length",
        max_length=16000 * 4,
        truncation=True,
        return_tensors="pt"
    )

    valid_labels = [batch["label"][i] for i in valid_indices]

    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": torch.tensor(valid_labels)
    }


# Process test dataset
print("\nPreprocessing test data...")
processed_test = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=test_dataset.column_names
)

# Filter out None results
processed_test = processed_test.filter(lambda x: x is not None)
print(f"After preprocessing, {len(processed_test)} samples remaining")

# Evaluation setup
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Initialize Trainer
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

# Run evaluation
print("\nRunning evaluation...")
results = trainer.evaluate(processed_test)
print(f"Evaluation results: {results}")

# Generate predictions
print("\nGenerating predictions...")
predictions = trainer.predict(processed_test)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Classification report
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()