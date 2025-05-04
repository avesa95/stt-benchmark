from abc import ABC, abstractmethod
import numpy as np
import wave
import subprocess
import sys
import json

import torch
import whisperx
from pywhispercpp.model import Model as WhisperCppModel
from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# Base STT Model class
class STTModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def transcribe(self, audio_tensor: torch.Tensor) -> str:
        pass


# Whisper (transformers) model wrapper
class WhisperOpenAIModel(STTModel):
    def __init__(self, model_id="openai/whisper-tiny", device="cpu"):
        super().__init__(name=model_id.split("/")[-1])
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            device
        )
        self.device = device
        self.model.eval()

    def transcribe(self, audio_tensor: torch.Tensor) -> str:
        input_features = self.processor(
            audio_tensor.squeeze(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[
            0
        ].strip()


# Placeholder: whisper.cpp model (run subprocess or binding)
class WhisperCpp(STTModel):
    def __init__(self, model_path):
        super().__init__(name="whisper.cpp")
        self.model = WhisperCppModel(model_path)

    def transcribe(self, audio_path: str, print_progress: bool = False) -> str:
        segments = self.model.transcribe(
            audio_path,
            print_progress=print_progress,
        )
        # Join all segment texts into a single string
        return " ".join(segment.text for segment in segments).strip()


class WhisperX(STTModel):
    def __init__(
        self, 
        model_id: str = "tiny", 
        device: str = "cpu", 
        compute_type: str = "int8"
    ):
        super().__init__(name=model_id)
        self.model = whisperx.load_model(model_id, device, compute_type=compute_type)

    def transcribe(
        self, 
        audio: np.ndarray, 
        batch_size: int = 4
    ) -> str:
        result = self.model.transcribe(audio, batch_size=batch_size)
        return " ".join(segment.get("text") for segment in result["segments"]).strip()


class Vosk(STTModel):
    
    def __init__(self, model_id: str = "vosk-model-small-en-us-0.15"):
        super().__init__(name=model_id)
        self.model = VoskModel(model_name=model_id)

    def transcribe(
        self, 
        model_path: str,
        sample_rate: int = 16000
    ) -> str:
        
        SetLogLevel(0)

        # wf = wave.open(model_path, "rb")
        # if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        #     raise ValueError("Audio file must be WAV format mono PCM.")

        rec = KaldiRecognizer(self.model, sample_rate)
        # rec.SetWords(True)
        # rec.SetPartialWords(True)


        with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i", model_path,
                            "-ar", str(sample_rate) , "-ac", "1", "-f", "s16le", "-"],
                            stdout=subprocess.PIPE) as process:

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
    model = Vosk("vosk-model-small-en-us-0.15")

    # audio = whisperx.load_audio("/Users/ristoc/Workspaces/cube/stt-benchmark/data/M18_05_01.wav")
    print(model.transcribe("/Users/ristoc/Workspaces/cube/stt-benchmark/data/M18_05_01.wav"))