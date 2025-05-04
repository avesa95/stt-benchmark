from abc import ABC, abstractmethod


# Base SLM Model class
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
    def __init__(self, temperature: float = 0.3):
        super().__init__(model_name="qwen", temperature=temperature)


class GemmaModel(OllamaSLM):
    def __init__(self, temperature: float = 0.3):
        super().__init__(model_name="gemma", temperature=temperature)


if __name__ == "__main__":
    slm = QwenModel()
    print("Qwen output:\n", slm.complete("Fix the grammar: he go to school every day"))

    slm = GemmaModel()
    print(
        "Gemma output:\n",
        slm.complete("Remove the filler words: uh I guess we can maybe go now"),
    )
