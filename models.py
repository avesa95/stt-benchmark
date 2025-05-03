from abc import ABC, abstractmethod

import torch
from pywhispercpp.model import Model as WhisperCppModel
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
