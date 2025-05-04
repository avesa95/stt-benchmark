import logging

import nltk
import pandas as pd
from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm

from src.slm.models import QwenModel


def format_prompt(incorrect_sentence: str) -> str:
    few_shot_examples = [
        {
            "input": "He go to store every day and buy stuffs he need.",
            "output": "He goes to the store every day and buys the things he needs.",
        },
        {
            "input": "Umm I guess maybe we should like leave now or something.",
            "output": "I guess we should leave now.",
        },
        {
            "input": "She don’t know what she wanna do after graduation.",
            "output": "She doesn't know what she wants to do after graduation.",
        },
    ]

    instructions = (
        "You are a grammar correction assistant.\n"
        "Correct the sentence by:\n"
        "- Fixing grammar and punctuation\n"
        "- Removing filler words (e.g. 'umm', 'like', 'you know')\n"
        "- Making the sentence clear and concise\n"
        "- Splitting run-on sentences if needed\n\n"
        "Return only the corrected sentence. Do not explain the steps.\n"
        "Do not repeat the instructions or say 'Sure'.\n\n"
        "Give your response in the same language as the sentence.\n\n"
    )

    # Add few-shot examples
    examples_text = ""
    for ex in few_shot_examples:
        examples_text += f"Input: {ex['input']}\nOutput: {ex['output']}\n---\n"

    # Final prompt
    prompt = f"{instructions}\n{examples_text}Input: {incorrect_sentence}\nOutput:"
    return prompt


def evaluate_prediction(predicted: str, reference: str):
    smoothie = SmoothingFunction().method4
    reference_tokens = nltk.word_tokenize(reference.lower())
    predicted_tokens = nltk.word_tokenize(predicted.lower())

    bleu_score = sentence_bleu(
        [reference_tokens], predicted_tokens, smoothing_function=smoothie
    )
    word_error = wer(reference, predicted)
    char_error = cer(reference, predicted)
    return bleu_score, word_error, char_error


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    df = pd.read_csv(
        "/Users/vesaalexandru/Workspaces/cube/RealTyme/stt-benchmark/src/slm/datasets/Long_Sentence_Grammar_Correction_Dataset.csv"
    )
    model = QwenModel(temperature=0.1)
    results = []

    logging.info("🔍 Benchmarking Gemma on long spoken-style corrections...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = format_prompt(row["incorrect"])
        try:
            output = model.complete(prompt)
        except Exception as e:
            logging.error(f"Error during model completion at row {idx}: {e}")
            output = ""
        bleu, word_error, char_error = evaluate_prediction(output, row["corrected"])
        results.append(
            {
                "category": row["category"],
                "incorrect": row["incorrect"],
                "expected": row["corrected"],
                "model_output": output,
                "bleu": round(bleu, 4),
                "wer": round(word_error, 4),
                "cer": round(char_error, 4),
            }
        )
        if idx % 10 == 0:
            logging.info(f"Processed {idx + 1}/{len(df)} rows...")

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        "/Users/vesaalexandru/Workspaces/cube/RealTyme/stt-benchmark/src/slm/grammatical_error_correction/evals/gemma_long_sentence_benchmark.csv",
        index=False,
    )

    avg_bleu = result_df["bleu"].mean()
    avg_wer = result_df["wer"].mean()
    avg_cer = result_df["cer"].mean()

    logging.info("✅ Benchmark complete!")
    logging.info(f"🔹 Average BLEU: {avg_bleu:.3f}")
    logging.info(f"🔹 Average WER: {avg_wer:.3f}")
    logging.info(f"🔹 Average CER: {avg_cer:.3f}")
    logging.info("📄 Full report saved to gemma_long_sentence_benchmark.csv")


if __name__ == "__main__":
    main()
