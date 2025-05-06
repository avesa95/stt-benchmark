import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod

import torch
import torchaudio
import whisperx
from pywhispercpp.model import Model as WhisperCppModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from vosk import KaldiRecognizer, SetLogLevel
from vosk import Model as VoskModel


# Base STT Model class
class STTModel(ABC):
    def __init__(self, name):
        self.name = name
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass

    def _preprocess_audio(self, audio_path: str) -> str:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != 16000:
            waveform = self.resampler(waveform)

        return waveform.numpy()


# Whisper (transformers) model wrapper
class WhisperOpenAIModel(STTModel):
    def __init__(self, model_id="openai/whisper-tiny", device="cpu"):
        super().__init__(name=model_id.split("/")[-1])
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            device
        )
        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        self.device = device
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = self.resampler(waveform)

        # Mono channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        input_features = self.processor(
            waveform.squeeze(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ].strip()
    

class WhisperCpp(STTModel):
    def __init__(self, model_path, language="auto"):
        super().__init__(name="whisper.cpp")
        self.model_name = model_path
        self.language = language

    def transcribe(self, audio_path: str):
        """
        Processes an audio file using a specified model and returns the processed string.

        :param wav_file: Path to the WAV file
        :param model_name: Name of the model to use
        :return: Processed string output from the audio processing
        :raises: Exception if an error occurs during processing
        """

        # Check if the file exists
        if not os.path.exists(self.model_name):
            raise FileNotFoundError(f"Model file not found: {self.model_name} \n\nDownload a model with this command:\n\n> bash ./models/download-ggml-model.sh {self.model_name}\n\n")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"WAV file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = self.resampler(waveform)

        # Save as 16-bit PCM WAV for whisper.cpp
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            processed_audio_path = tmp.name
            torchaudio.save(
                processed_audio_path,
                waveform,
                16000,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            full_command = f"/Users/ristoc/Workspaces/cube/stt-benchmark/whisper.cpp/build/bin/whisper-cli -m {self.model_name} -f {processed_audio_path} -l {self.language} -nt"

            # Execute the command
            process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Get the output and error (if any)
            output, error = process.communicate()
            # print(error)

            # if error:
            #     raise Exception(f"Error processing audio: {error.decode('utf-8')}")

            # Process and return the output string
            decoded_str = output.decode('utf-8').strip()
            processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()

            return processed_str


# Placeholder: whisper.cpp model (run subprocess or binding)
class PyWhisperCpp(STTModel):
    def __init__(self, model_path, language="auto"):
        super().__init__(name="whisper.cpp")
        self.model = WhisperCppModel(model_path, split_on_word=True, translate=False)
        self.language = language

    def transcribe(self, audio_path: str, print_progress: bool = False) -> str:

        segments = self.model.transcribe(
            audio_path,
            print_progress=print_progress,
            language=self.language,
        )
        # Join all segment texts into a single string
        return " ".join(segment.text for segment in segments).strip()


class WhisperX(STTModel):
    def __init__(
        self,
        model_id: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ):
        super().__init__(name=model_id)
        self.model = whisperx.load_model(
            model_id, device, compute_type=compute_type, language=language
        )

    def transcribe(self, audio_path: str, batch_size: int = 4) -> str:
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=batch_size)
        return " ".join(segment.get("text") for segment in result["segments"]).strip()


class Vosk(STTModel):
    model_per_language = {
        "en": "vosk-model-small-en-us-0.15",
        "fr": "vosk-model-small-fr-0.22",
        "ar": "vosk-model-ar-mgb2-0.4",
        "de": "vosk-model-small-de-0.15",
        "it": "vosk-model-small-it-0.22",
        "es": "vosk-model-small-es-0.42",
    }

    def __init__(self, language: str = "en"):
        model_id = self.model_per_language[language]
        super().__init__(name=model_id)
        self.model = VoskModel(model_name=model_id)

    def transcribe(self, model_path: str, sample_rate: int = 16000) -> str:
        SetLogLevel(0)

        # wf = wave.open(model_path, "rb")
        # if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        #     raise ValueError("Audio file must be WAV format mono PCM.")

        rec = KaldiRecognizer(self.model, sample_rate)
        # rec.SetWords(True)
        # rec.SetPartialWords(True)

        with subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-i",
                model_path,
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-f",
                "s16le",
                "-",
            ],
            stdout=subprocess.PIPE,
        ) as process:
            while True:
                data = process.stdout.read(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    rec.Result()
                else:
                    rec.PartialResult()

            result = json.loads(rec.FinalResult())
            return result.get("text", "")


if __name__ == "__main__":
    # model = WhisperCpp("tiny", "en")
    model = WhisperCppRaw("/Users/ristoc/Workspaces/cube/stt-benchmark/whisper.cpp/models/ggml-medium-q8_0.bin", "en")
    print(model.transcribe("/Users/ristoc/Workspaces/cube/stt-benchmark/data/M18_05_01.wav"))

    # # audio = whisperx.load_audio("/Users/ristoc/Workspaces/cube/stt-benchmark/data/M18_05_01.wav")
    # print(model.transcribe_with_rtf("/Users/ristoc/Workspaces/cube/stt-benchmark/data/M18_05_01.wav"))
