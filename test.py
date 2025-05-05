import csv
import time
from datetime import datetime
from math import floor

import ffmpeg
from jiwer import cer, wer
from tqdm import tqdm

from src.stt.dataset import get_samples_per_language
from src.stt.models import WhisperCpp

# Load datasets
# languages = ["en", "fr", "ar", "de", "it", "es"]
languages = ["en"]

samples_per_lang = get_samples_per_language(languages=languages, num_samples=10)

# # Load Whisper model and processor (use base for benchmarking, adjust for tiny or others)
# model_name = "openai/whisper-base"
# processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)
# model.eval()

# # Prepare device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

language_names = {
    "en": "English",
    "fr": "French",
    "ar": "Arabic",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
}

results = {}

for lang_code, samples in samples_per_lang.items():
    model = WhisperCpp("tiny.en", language=lang_code)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"test/whisper_cpp_{lang_code}_benchmark_results_{timestamp}.csv"

    # model = WhisperX(language=lang_code)

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Ground Truth",
            "Transcription",
            "WER",
            "CER",
            "Processing Time",
            "Audio Duration",
            "RTF",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        shuffled_samples = samples.shuffle(seed=42)
        print(f"\nBenchmarking {lang_code} over {len(samples)} samples...")

        language_wer = []
        language_rtf = []
        language_cer = []

        for sample in tqdm(shuffled_samples):
            audio_path = sample["path"]
            ground_truth = sample["sentence"]

            probe = ffmpeg.probe(audio_path)
            audio_duration = floor(float(probe["format"]["duration"]))

            start_time = time.time()
            transcription = model.transcribe(audio_path)
            processing_time = time.time() - start_time

            # Evaluate
            word_error = wer(ground_truth.lower(), transcription.lower())
            char_error = cer(ground_truth.lower(), transcription.lower())
            language_wer.append(word_error)
            language_rtf.append(processing_time / audio_duration)
            language_cer.append(char_error)

            writer.writerow(
                {
                    "Ground Truth": ground_truth,
                    "Transcription": transcription,
                    "WER": word_error,
                    "CER": char_error,
                    "Processing Time": processing_time,
                    "Audio Duration": audio_duration,
                    "RTF": processing_time / audio_duration,
                }
            )

        print(f"\nResults saved to {csv_filename}")

    avg_wer = sum(language_wer) / len(language_wer)
    avg_rtf = sum(language_rtf) / len(language_rtf)
    results[lang_code] = {
        "language": language_names[lang_code],
        "avg_wer": avg_wer,
        "avg_cer": sum(language_cer) / len(language_cer),
        "avg_rtf": avg_rtf,
    }

print(results)
