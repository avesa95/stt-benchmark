from abc import ABC, abstractmethod
import ollama

import ollama


class SLMModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass


class OllamaSLM(SLMModel):
    def __init__(self, model_name: str = "qwen", temperature: float = 0.3):
        super().__init__(name=f"ollama-{model_name}")
        self.model_name = model_name
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"[ERROR] {e}"


class QwenModel(OllamaSLM):
    def __init__(self, temperature: float = 0.3, model_name: str = "qwen:4b"):
        super().__init__(model_name=model_name, temperature=temperature)


class GemmaModel(OllamaSLM):
    def __init__(self, temperature: float = 0.3):
        super().__init__(model_name="gemma:2b", temperature=temperature)


if __name__ == "__main__":
    slm = GemmaModel()
    print(
        "Gemma output:\n",
        slm.complete("Remove the filler words: uh I guess we can maybe go now"),
    )
