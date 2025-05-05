import logging

import nltk
import pandas as pd
from jiwer import cer, wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from tqdm import tqdm

from src.slm.models import GemmaModel, QwenModel
from src.slm.prompts.prompt_generator import PromptGenerator


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


def run_benchmark(
    df: pd.DataFrame, model, prompt_generator: PromptGenerator, language: str
) -> pd.DataFrame:
    results = []

    logging.info("🔍 Starting benchmark...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = prompt_generator.generate(language, row["incorrect"])
        try:
            output = model.complete(prompt)
        except Exception as e:
            logging.error(f"Error during model completion at row {idx}: {e}")
            output = ""

        scores = evaluate_prediction(output, row["corrected"])
        results.append(
            {
                "category": row["category"],
                "incorrect": row["incorrect"],
                "expected": row["corrected"],
                "model_output": output,
                "bleu": round(scores["BLEU"], 4),
                "gleu": round(scores["GLEU"], 4),
                "wer": round(scores["WER"], 4),
                "cer": round(scores["CER"], 4),
            }
        )
        if idx % 10 == 0:
            logging.info(f"Processed {idx + 1}/{len(df)} rows...")

    return pd.DataFrame(results)


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    dataset_path = "/Users/vesaalexandru/Workspaces/cube/RealTyme/stt-benchmark/src/slm/datasets/Long_Sentence_Grammar_Correction_Dataset.csv"
    output_path = (
        "/Users/vesaalexandru/Workspaces/cube/RealTyme/stt-benchmark/src/slm/grammatical_error_correction/evals"
        "/gemma_english_long_sentence_grammar_correction_benchmark.csv"
    )

    language = "en"

    df = pd.read_csv(dataset_path)
    qwen_model = QwenModel(temperature=0.03, model_name="qwen:4b")
    gemma_model = GemmaModel(temperature=0.03, model_name="gemma:2b")
    prompt_generator = PromptGenerator()

    result_df = run_benchmark(
        df, model=qwen_model, prompt_generator=prompt_generator, language=language
    )
    result_df.to_csv(output_path, index=False)

    logging.info("✅ Benchmark complete!")
    logging.info(f"🔹 Average BLEU: {result_df['bleu'].mean():.3f}")
    logging.info(f"🔹 Average GLEU: {result_df['gleu'].mean():.3f}")
    logging.info(f"🔹 Average WER: {result_df['wer'].mean():.3f}")
    logging.info(f"🔹 Average CER: {result_df['cer'].mean():.3f}")
    logging.info(f"📄 Full report saved to {output_path}")


if __name__ == "__main__":
    main()
