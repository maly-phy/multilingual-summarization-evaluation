from aggregate_eval_scores import aggregate_scores, save_scores
import pandas as pd
import os
import numpy as np


def compute_lci_score(df_rej_eng, df_acc_eng, df_rej_ger, df_acc_ger, file_name):
    if "LAR" in df_rej_eng.columns:
        df_rej_eng.rename(
            columns={"LAR_mean": "lar_score_mean", "LAR_std": "lar_score_std"},
            inplace=True,
        )
        df_rej_ger.rename(
            columns={"LAR_mean": "lar_score_mean", "LAR_std": "lar_score_std"},
            inplace=True,
        )

    df_rej_eng.columns = df_rej_eng.columns.str.replace("_base", "", regex=False)
    df_rej_ger.columns = df_rej_ger.columns.str.replace("_base", "", regex=False)

    agg_dict = {}
    for metric in list(df_rej_eng.columns):
        if "mean" not in metric:
            continue
        if "rouge" in metric and "mean" in metric and not "f1" in metric:
            continue

        rej_score_eng = df_rej_eng[metric]
        acc_score_eng = df_acc_eng[metric]
        delta_score_eng = abs(acc_score_eng - rej_score_eng)

        rej_score_ger = df_rej_ger[metric]
        acc_score_ger = df_acc_ger[metric]
        delta_score_ger = abs(acc_score_ger - rej_score_ger)
        deno = 2 * np.maximum(delta_score_eng, delta_score_ger)
        lci_score = np.where(deno == 0, 0.0, (delta_score_eng + delta_score_ger) / deno)
        agg_dict[f"{metric.replace('_mean', '')}_lci"] = lci_score[0]

    out_df = pd.DataFrame([agg_dict], index=[0])
    save_dir = "evaluation/single_metrics/consistency_eval"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    out_df.to_csv(file_path, index=False)

    txt_file = file_name.replace("consistency", "agg").replace(".csv", ".txt")
    txt_path = os.path.join(save_dir, txt_file)
    save_scores(txt_path, out_df, "w")

    return out_df


if __name__ == "__main__":
    global_task = "summary_criteria_eval"
    task = "combined_eval"
    file_name = "criteria_consistency_scores.csv"

    file_path_rej_eng = f"evaluation/English/{global_task}/rej_{task}.csv"
    file_path_acc_eng = f"evaluation/English/{global_task}/acc_{task}.csv"
    file_path_rej_ger = f"evaluation/German/{global_task}/rej_{task}.csv"
    file_path_acc_ger = f"evaluation/German/{global_task}/acc_{task}.csv"
    df_rej_eng = pd.read_csv(file_path_rej_eng)
    df_acc_eng = pd.read_csv(file_path_acc_eng)
    df_rej_ger = pd.read_csv(file_path_rej_ger)
    df_acc_ger = pd.read_csv(file_path_acc_ger)
    out_df = compute_lci_score(
        df_rej_eng, df_acc_eng, df_rej_ger, df_acc_ger, file_name
    )
    print(out_df.shape)
