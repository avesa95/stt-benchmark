import torch
import torchaudio
from jiwer import wer
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset import get_samples_per_language

# Load datasets
languages = ["en", "fr", "ar"]


samples_per_lang = get_samples_per_language(languages=["fr"])

# Load Whisper model and processor (use base for benchmarking, adjust for tiny or others)
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Prepare device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Resampler to convert to 16kHz mono
resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

language_names = {"en": "English", "fr": "French", "ar": "Arabic"}

results = {}

for lang_code, samples in samples_per_lang.items():
    print(f"\nBenchmarking {lang_code}...")
    language_wer = []

    for sample in tqdm(samples):
        audio_path = sample["path"]
        ground_truth = sample["sentence"]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = resampler(waveform)

        # Mono channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Tokenize and run Whisper
        input_features = processor(
            waveform.squeeze(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ].strip()

        # Evaluate
        error = wer(ground_truth.lower(), transcription.lower())
        language_wer.append(error)

    avg_wer = sum(language_wer) / len(language_wer)
    results[lang_code] = {"language": language_names[lang_code], "avg_wer": avg_wer}

print(results)
