from src.slm.prompts.english import english_grammar_correction_prompt
from src.slm.prompts.french import french_grammar_correction_prompt


class PromptGenerator:
    def __init__(self):
        self.prompts = {
            "en": self._english_prompt,
            "fr": self._french_prompt,
        }

    def generate(self, language: str, incorrect_sentence: str) -> str:
        if language not in self.prompts:
            raise ValueError(f"Unsupported language: {language}")
        return self.prompts[language](incorrect_sentence)

    def _english_prompt(self, incorrect_sentence: str) -> str:
        # your full english_grammar_correction_prompt here
        return english_grammar_correction_prompt(incorrect_sentence)

    def _french_prompt(self, incorrect_sentence: str) -> str:
        # your full french_grammar_correction_prompt here
        return french_grammar_correction_prompt(incorrect_sentence)
