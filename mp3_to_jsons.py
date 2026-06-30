import os
import json
from faster_whisper import WhisperModel

# ----------------------------------------------------
# Fix OpenMP issue on Windows
# ----------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------------------------------
# Load Faster Whisper Model
# ----------------------------------------------------
model = WhisperModel(
    "distil-large-v3",
    device="cpu",
    compute_type="int8"
)

# ----------------------------------------------------
# Create output folder if it doesn't exist
# ----------------------------------------------------
os.makedirs("json", exist_ok=True)

# ----------------------------------------------------
# Read all audio files
# ----------------------------------------------------
audios = os.listdir("audios")

for audio in audios:

    if "_" not in audio:
        continue

    number = audio.split("_")[0]

    title = os.path.splitext(audio.split("_", 1)[1])[0]

    audio_path = os.path.join("audios", audio)

    print(f"\nProcessing : {audio}")

    segments, info = model.transcribe(
        audio_path,
        language="hi",
        task="transcribe",
        beam_size=3
    )

    chunks = []
    full_text = ""

    for segment in segments:

        text = segment.text.strip()

        chunks.append({
            "number": number,
            "title": title,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text
        })

        full_text += text + " "

    data = {
        "number": number,
        "title": title,
        "language": "Hindi",
        "transcript": full_text.strip(),
        "chunks": chunks
    }

    output_file = os.path.join(
        "json",
        f"{os.path.splitext(audio)[0]}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"Saved -> {output_file}")

print("\nAll files processed successfully!")