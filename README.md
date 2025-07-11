
# 🎙️ Emotion Detection from Raw Audio Waveform

This mini project is designed to detect human emotions such as **happy**, **sad**, **angry**, **neutral**, etc., directly from raw audio input (i.e., `.wav` files). It uses a **pretrained deep learning model** from Hugging Face to classify emotions based on audio waveforms without converting them into text.

---

## 📌 Features

- 🎧 Accepts `.wav` audio files (raw speech)
- 🧠 Uses state-of-the-art pretrained transformer models like `HuBERT`
- 💬 Classifies emotions from voice: Happy, Sad, Angry, Neutral, etc.
- ⚡ Real-time inference and web demo support
- 📦 Built with Python, Hugging Face Transformers, Torchaudio, and PyTorch

---

## 🔍 How It Works

### 1. Audio Input
- User provides a raw audio file (`.wav`).
- Ideal format: **Mono**, **16-bit PCM**, **16kHz sampling rate**.

### 2. Preprocessing
- Resampling to 16kHz if needed
- Normalizing volume (optional)
- Converting audio into PyTorch tensor

### 3. Feature Extraction & Inference
- A **pretrained model** like `superb/hubert-large-superb-er` from Hugging Face is used.
- The model extracts acoustic features and passes them through a classification head.
- Output: Predicted emotion label with probability scores.

### 4. Output
- Emotion prediction is displayed with confidence.
- Optionally integrated with Gradio for live demos.

---

## 📁 Project Directory Structure

```
emotion_detection/
├── audio_samples/              # Sample audio files for testing
├── model/
│   └── emotion_model.py        # Model loading and inference logic
├── utils/
│   └── audio_utils.py          # Audio preprocessing utilities
├── app/
│   ├── main.py                 # Main Python script for CLI use
│   └── interface.py            # Gradio interface (optional)
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/emotion-detection-audio.git
cd emotion-detection-audio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Emotion Detection from CLI
```bash
python app/main.py --file audio_samples/sample.wav
```

### 4. Run Web Interface (Gradio)
```bash
python app/interface.py
```

---

## 🧠 Model Explanation

### What is HuBERT?
- **HuBERT (Hidden-Unit BERT)** is a self-supervised speech representation model.
- Learns audio patterns and structure from unlabeled data.
- Pretrained on massive datasets and fine-tuned for emotion classification.

### Why Raw Waveform?
- Avoids loss of information due to feature engineering (like MFCCs).
- Enables model to learn directly from raw audio, increasing emotion accuracy.

### Emotions Detected
- Neutral 😐
- Happy 😊
- Angry 😡
- Sad 😢
- Others (depending on model support)

---

## 💡 Example Output

```bash
Input: audio_samples/happy.wav
Predicted Emotion: Happy 😄
Confidence: 94.8%
```

---

## ✅ Requirements

- Python 3.8+
- torch
- torchaudio
- transformers (Hugging Face)
- gradio (optional for UI)

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 📦 Models Used

| Model Name | Link |
|------------|------|
| `superb/hubert-large-superb-er` | [Hugging Face Model](https://huggingface.co/superb/hubert-large-superb-er) |

---

## 🌍 Use Cases

- Mental Health Monitoring
- Emotion-aware Virtual Assistants
- Call Center Quality Analysis
- Language-independent Sentiment Recognition

---

## ✍️ Author

**Dhanesh J.**  
This is a mini project developed as part of a Computer Science curriculum. It explores the integration of AI and speech processing for understanding human emotions.

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🚀 Future Work

- 🎤 Real-time microphone input support
- 🌐 Multilingual emotion detection
- 📊 Advanced dashboard with emotion statistics
