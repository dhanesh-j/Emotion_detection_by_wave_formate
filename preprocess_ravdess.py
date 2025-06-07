import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path  # Better path handling for Windows

# Emotion mapping based on filename
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def create_dataset_csv(root_dir, output_csv='ravdess_data.csv'):
    data = []
    skipped_files = 0
    root_path = Path(root_dir)  # Convert to Path object

    # Get all actor folders
    actors = [d for d in os.listdir(root_path) if (root_path / d).is_dir()]

    for actor in tqdm(actors, desc="Processing actors"):
        actor_path = root_path / actor
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                try:
                    parts = file.split('-')
                    if len(parts) < 3:
                        skipped_files += 1
                        continue

                    emotion_code = parts[2]
                    emotion = emotion_map.get(emotion_code)

                    if not emotion:
                        skipped_files += 1
                        continue

                    file_path = actor_path / file
                    if not file_path.exists():
                        skipped_files += 1
                        continue

                    # Convert Path to string for CSV
                    data.append([str(file_path), emotion])
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    skipped_files += 1

    # Save to CSV
    df = pd.DataFrame(data, columns=['path', 'emotion'])
    df.to_csv(output_csv, index=False)

    print(f"✅ Dataset CSV saved as {output_csv}")
    print(f"Total samples: {len(df)}")
    print(f"Skipped files: {skipped_files}")

    return df

if __name__ == "__main__":
    # Use raw string literal for Windows paths
    root_dir = r'C:\Users\jayac\PycharmProjects\PythonProject\dataset\Audio_Speech_Actors_01-24'
    create_dataset_csv(root_dir)