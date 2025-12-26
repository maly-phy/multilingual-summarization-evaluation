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
    # print(df.head(2), "\n")
    print(df.info(), "\n")
    print(df.select_dtypes(include=["number"]).columns, "\n")
    return df


def aggregate_scores(file_path, language):
    df = pd.read_csv(file_path)[:30]
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


def process_files(language, global_task, tasks):
    scores_file = f"evaluation/{language}/{global_task}/agg_scores.txt"
    all_dfs = []
    for task in tasks:
        file_path = f"evaluation/{language}/{global_task}/{task}_eval_0_29.csv"
        agg_df = aggregate_scores(file_path, language)
        all_dfs.append(agg_df)
        append_scores_to_file(scores_file, agg_df)

    combined_df = pd.concat(all_dfs, axis=1, ignore_index=False)
    combined_df.to_csv(
        f"evaluation/{language}/{global_task}/combined_eval.csv", index=False
    )


if __name__ == "__main__":
    tasks = [
        "semantic",
        "blanc_estim",
        "bleurt",
        "hf",
        "lens",
    ]
    language = "German"
    global_task = "nlp_eval_facts"
    file_path = f"evaluation/{language}/{global_task}/{tasks}_eval_0_29.csv"
    scores_file = f"evaluation/{language}/{global_task}/agg_scores.txt"
    # df = pd.read_csv(file_path)
    # report_df(file_path, language)
    process_files(language, global_task, tasks)
