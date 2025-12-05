from utils import extract_content_between_tags
import pandas as pd
import os

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 500)


def convert_confidence(value):
    if isinstance(value, str):
        value = value.replace("%", "")
    try:
        return float(value)
    except ValueError:
        return None


def preprocess_llm_scores(df):
    for col in df.columns:
        if "confidence" in col.lower():
            df[col] = df[col].apply(
                lambda x: extract_content_between_tags(x, "confidence")
            )
            df[col] = df[col].apply(convert_confidence)

        if "structure_score" in col.lower():
            df.loc[1:2, col] = df.loc[1:2, col].apply(
                lambda x: x.split("\n\n")[-1].strip()
            )
            df[col] = df[col].astype(int)

        if "dynamics_score" in col.lower():
            df[col] = (
                df[col.replace("Score", "Confidence")]
                .apply(lambda x: extract_content_between_tags(x, "score"))
                .astype(int)
            )

    return df


def clean_basic_meeting_eval(df):
    for col in df.columns:
        if "confidence" in col.lower():
            df[col] = df[col].apply(convert_confidence).astype(float)

    return df


if __name__ == "__main__":
    file_path = "evaluation/English/meeting_challenges_eval/llama-3.1-8b-instant_meeting_challenges_eval_0_19.csv"
    df = pd.read_csv(file_path)
    df = preprocess_llm_scores(df)

    print(df.dtypes)
