import nltk
from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.gleu_score import sentence_gleu

from src.slm.models import QwenModel
from src.slm.prompts.prompt_generator import PromptGenerator

nltk.download("punkt")


def evaluate_prediction(predicted: str, reference: str):
    smoothie = SmoothingFunction().method4
    reference_tokens = nltk.word_tokenize(reference.lower())
    predicted_tokens = nltk.word_tokenize(predicted.lower())

    bleu_score = sentence_bleu(
        [reference_tokens], predicted_tokens, smoothing_function=smoothie
    )
    gleu_score = sentence_gleu([reference_tokens], predicted_tokens)
    word_error = wer(reference, predicted)
    char_error = cer(reference, predicted)

    return {
        "BLEU": bleu_score,
        "GLEU": gleu_score,
        "WER": word_error,
        "CER": char_error,
    }


if __name__ == "__main__":
    prompt_generator = PromptGenerator()

    incorrect_example = """So I was kinda walk to the storesss and then I seessn this guy who I think he was in my math class but I'm not kinda super sure and then he just kind of looked at me and I didn’t really knowww what to do so I just kept dds, and umm then I remembered I forgot my wallet so I had to turn back."""
    model = QwenModel(temperature=0.1, model_name="qwen:4b")
    results = []

    prompt = prompt_generator.generate(
        language="en", incorrect_sentence=incorrect_example
    )
    output = model.complete(prompt)
    print(output)
    scores = evaluate_prediction(output, incorrect_example)

    print(f"BLEU: {scores['BLEU']}")
    print(f"GLEU: {scores['GLEU']}")
    print(f"WER: {scores['WER']}")
    print(f"CER: {scores['CER']}")
