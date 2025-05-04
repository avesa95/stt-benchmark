import os
from datasets import load_dataset



def get_samples_per_language(languages=None, num_samples=20, min_words=3):
    """
    Loads Common Voice samples for each language and returns a dictionary
    mapping language codes to filtered datasets.

    Args:
        languages (list or None): List of language codes to load (e.g., ['en', 'fr']).
                                  If None, defaults to ['en', 'fr', 'ar'].
        num_samples (int): Number of samples to select per language.
        min_words (int): Minimum number of words in a sentence.

    Returns:
        dict: {lang_code: filtered_dataset}
    """
    if languages is None:
        languages = ["en", "fr", "ar"]
    samples_per_lang = {}
    for lang_code in languages:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_13_0", lang_code, split="test", token=os.getenv("HUGGINGFACE_TOKEN")
        )
        filtered = dataset.filter(
            lambda x: len(x["sentence"].split()) > min_words
        ).select(range(num_samples))
        samples_per_lang[lang_code] = filtered
    return samples_per_lang


if __name__ == "__main__":
    samples_per_lang = get_samples_per_language(languages=["en", "fr", "de", "it", "ar", "es"])
    print(samples_per_lang)
