import pandas as pd
from deep_translator import GoogleTranslator


def transalte_dataset(
    df: pd.DataFrame, source_language: str, target_language: str, output_path: str
) -> pd.DataFrame:
    translator = GoogleTranslator(source=source_language, target=target_language)
    df_new = df[["category", "incorrect", "corrected"]].copy()
    df_new["incorrect"] = df_new["incorrect"].apply(lambda x: translator.translate(x))
    df_new["corrected"] = df_new["corrected"].apply(lambda x: translator.translate(x))

    df_new.to_csv(output_path, index=False)
    return df_new
