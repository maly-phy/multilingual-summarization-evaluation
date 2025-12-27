import pandas as pd
import os
from clean_score_results import preprocess_llm_scores, clean_basic_meeting_eval

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 500)


def report_df(file_path, language):
    df = pd.read_csv(file_path)
    df = clean_basic_meeting_eval(df)
    if "meeting_challenges" in file_path:
        df = preprocess_llm_scores(df, language)

    print(df.shape, "\n")
    print(df.columns, "\n")
    # print(df.head(2), "\n")
    print(df.info(), "\n")
    print(df.select_dtypes(include=["number"]).columns, "\n")
    return df


def aggregate_scores(file_path, language):
    df = pd.read_csv(file_path)[:30]
    df = clean_basic_meeting_eval(df)

    if "meeting_challenges" in file_path:
        df = preprocess_llm_scores(df, language)

    if language == "English" and "rej_factuality" in file_path:
        df = df.drop(index=[7]).reset_index(drop=True)
        df["Factuality_score"] = df["Factuality_score"].astype(float)

    agg_dict = {}
    for col in df.select_dtypes(include=["number"]).columns:
        agg_dict[f"{col}_mean"] = df[col].mean()
        agg_dict[f"{col}_std"] = df[col].std()

    agg_df = pd.DataFrame([agg_dict], index=[0])
    return agg_df


def save_scores(file_path, agg_df, save_type="w"):
    with open(file_path, save_type) as f:
        for k, v in agg_df.items():
            f.write(f"{k}: {v[0]}\n\n")


def process_files(language, global_task, tasks, type="", save_type="a"):
    scores_file = f"evaluation/{language}/{global_task}/{type}_agg_scores.txt"
    all_dfs = []
    for task in tasks:
        file_path = f"evaluation/{language}/summary_criteria_eval/model_summary_{type}_{task}_0_29.csv"
        if task == "factuality":
            file_path = file_path.replace("model", "ref")
        agg_df = aggregate_scores(file_path, language)
        all_dfs.append(agg_df)
        save_scores(scores_file, agg_df, save_type)

    combined_df = pd.concat(all_dfs, axis=1, ignore_index=False)
    combined_df.to_csv(
        f"evaluation/{language}/{global_task}/{type}_combined_eval.csv", index=False
    )


if __name__ == "__main__":
    language = "German"
    global_task = "summary_criteria_eval"
    tasks = ["linguistic", "naturalness", "factuality"]
    process_files(language, global_task, tasks, "acc", "a")
    exit(1)

    file_path = f"evaluation/{language}/summary_criteria_eval/ref_summary_rej_factuality_0_29.csv"
    df = pd.read_csv(file_path)
    df = report_df(file_path, language)
    print(df.shape)
