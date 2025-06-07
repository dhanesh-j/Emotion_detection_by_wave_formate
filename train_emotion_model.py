import pandas as pd
import torch
import torch_directml
import torchaudio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
import warnings
import os

# Initialize AMD GPU
try:
    device = torch_directml.device()
    print("AMD GPU detected - Using DirectML backend")
except Exception as error:
    device = torch.device("cpu")
    print(f"AMD GPU not available - Falling back to CPU. Error: {str(error)}")

# Suppress warnings
warnings.filterwarnings("ignore")

# Set torchaudio backend
torchaudio.set_audio_backend("soundfile")

# Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv('ravdess_data.csv')
    labels = sorted(df['emotion'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}  # Fixed typo in dict comprehension
    df['label'] = df['emotion'].map(label2id)
    dataset = Dataset.from_pandas(df)
    print(f"Loaded {len(df)} samples with {len(labels)} emotion classes")  # Fixed printf to print
except Exception as error:
    raise ValueError(f"Data loading error: {str(error)}")

# Model setup
print("Loading model...")
try:
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-base",
        return_attention_mask=True
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    ).to(device)
    print("Model loaded successfully")
except Exception as error:
    raise ValueError(f"Error loading model: {str(error)}")


def preprocess_function(batch):
    audio_arrays = []
    valid_indices = []

    for idx, path in enumerate(batch["path"]):  # Changed variable name from i to idx
        try:
            waveform, sample_rate = torchaudio.load(path)  # Changed variable name from sr to sample_rate
            waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            audio_arrays.append(waveform.numpy()[0])
            valid_indices.append(idx)
        except Exception as error:  # Changed variable name from e to error
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


# Process dataset
print("\nPreprocessing dataset...")
try:
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )

    dataset = dataset.filter(lambda x: x is not None)
    print(f"Final dataset size: {len(dataset)}")

    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

except Exception as error:
    raise ValueError(f"Dataset processing error: {str(error)}")

# Training configuration
training_args = TrainingArguments(
    output_dir="./wav2vec2-emotion",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=False,
    gradient_accumulation_steps=2,
    dataloader_pin_memory=True,
    save_safetensors=True
)


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    return {"accuracy": np.mean(predictions == eval_pred.label_ids)}


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
print("\nStarting training...")
try:
    trainer.train()
    print("\n✅ Training complete")

    output_dir = "final_emotion_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

except Exception as error:
    print(f"\n❌ Training failed: {str(error)}")
    raise