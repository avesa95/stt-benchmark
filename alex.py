import logging
import os
import tempfile
import torchaudio
from jiwer import wer
from tqdm import tqdm
from dataset import get_samples_per_language
from models import WhisperCpp  # Import your model abstraction
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
class STTBenchmark:
    def __init__(self, stt_model, language_names=None):
        self.stt_model = stt_model
        self.language_names = language_names or {
            "en": "English",
            "fr": "French",
            "ar": "Arabic",
        }
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
    def preprocess_audio(self, audio_path):
        logging.debug(f"Loading and preprocessing audio: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = self.resampler(waveform)
        # Save as 16-bit PCM WAV for whisper.cpp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            processed_audio_path = tmp.name
        torchaudio.save(
            processed_audio_path, waveform, 16000, encoding="PCM_S", bits_per_sample=16
        )
        logging.debug(f"Saved processed audio: {processed_audio_path}")
        return processed_audio_path
    def benchmark(self, samples_per_lang):
        results = {}
        for lang_code, samples in samples_per_lang.items():
            logging.info(
                f"Benchmarking language: {lang_code} "
                f"({self.language_names.get(lang_code, lang_code)})"
            )
            language_wer = []
            for sample in tqdm(samples):
                audio_path = sample["path"]
                ground_truth = sample["sentence"]
                # Check if model expects a file path or waveform
                if isinstance(self.stt_model, WhisperCpp):
                    processed_audio = self.preprocess_audio(audio_path)
                    logging.debug(f"Transcribing (WhisperCpp) file: {processed_audio}")
                    transcription = self.stt_model.transcribe(processed_audio)
                    # Clean up temporary file
                    os.unlink(processed_audio)
                else:
                    waveform = self.preprocess_audio(audio_path)
                    logging.debug(f"Transcribing (waveform) file: {audio_path}")
                    transcription = self.stt_model.transcribe(waveform)
                error = wer(ground_truth.lower(), transcription.lower())
                logging.debug(
                    f"GT: {ground_truth} | Pred: {transcription} | WER: {error:.3f}"
                )
                language_wer.append(error)
            avg_wer = sum(language_wer) / len(language_wer)
            logging.info(f"Avg WER for {lang_code}: {avg_wer:.3f}")
            results[lang_code] = {
                "language": self.language_names.get(lang_code, lang_code),
                "avg_wer": avg_wer,
            }
        return results