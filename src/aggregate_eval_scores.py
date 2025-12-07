import pandas as pd
import os
from clean_score_results import preprocess_llm_scores, clean_basic_meeting_eval

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 500)


def report_df(file_path, language):
    df = pd.read_csv(file_path)
    if "meeting_challenges" in file_path:
        df = preprocess_llm_scores(df, language)
    elif "basic_meeting" in file_path:
        df = clean_basic_meeting_eval(df)

    print(df.shape, "\n")
    print(df.columns, "\n")
    print(df.head(), "\n")
    print(df.info(), "\n")
    print(df.select_dtypes(include=["number"]).columns, "\n")
    return df


def aggregate_scores(file_path, language):
    df = pd.read_csv(file_path)
    if "meeting_challenges" in file_path:
        df = preprocess_llm_scores(df, language)
    elif "basic_meeting" in file_path:
        df = clean_basic_meeting_eval(df)

    agg_dict = {}
    for col in df.select_dtypes(include=["number"]).columns:
        agg_dict[f"{col}_mean"] = df[col].mean()
        agg_dict[f"{col}_std"] = df[col].std()

    agg_df = pd.DataFrame([agg_dict], index=[0])
    return agg_df


def save_scores(agg_df, language, task):
    save_dir = os.path.join("evaluation", language, f"{task}_eval")
    with open(os.path.join(save_dir, "agg_scores.txt"), "w") as f:
        for k, v in agg_df.items():
            f.write(f"{k}: {v[0]}\n\n")


def append_scores_to_file(file_path, agg_df):
    with open(file_path, "a") as f:
        for k, v in agg_df.items():
            f.write(f"{k}: {v[0]}\n\n")


if __name__ == "__main__":
    task = "summary_eval"
    language = "English"
    file_path = (
        f"evaluation/{language}/summary_eval/llama-3.1-8b-instant_{task}_with_LAR.csv"
    )
    scores_file = f"evaluation/{language}/summary_eval/agg_scores.txt"
    df = pd.read_csv(file_path)
    df["LAR_mean"] = df["LAR"].mean()
    df["LAR_std"] = df["LAR"].std()
    append_scores_to_file(scores_file, df[["LAR_mean", "LAR_std"]])
