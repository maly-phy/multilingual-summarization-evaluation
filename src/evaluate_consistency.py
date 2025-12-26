from aggregate_eval_scores import aggregate_scores
import pandas as pd
import os
import numpy as np


def compute_lci_score(df_rej_eng, df_acc_eng, df_rej_ger, df_acc_ger):
    agg_dict = {}
    for metric in list(df_acc_eng.columns):
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
        agg_dict[f"{metric.replace('_mean', '').replace('_score', '')}_lci"] = (
            lci_score[0]
        )

    out_df = pd.DataFrame([agg_dict], index=[0])
    print(f"out_df: {out_df.shape}")
    save_dir = "evaluation/single_metrics/consistency_eval"
    os.makedirs(save_dir, exist_ok=True)
    out_df.to_csv(os.path.join(save_dir, "consistency_eval.csv"), index=False)
    return out_df


if __name__ == "__main__":
    global_task = "nlp_eval"
    task = "combined_eval"
    file_path_rej_eng = f"evaluation/English/{global_task}/{task}.csv"
    file_path_acc_eng = f"evaluation/English/{global_task}_facts/{task}.csv"
    file_path_rej_ger = f"evaluation/German/{global_task}/{task}.csv"
    file_path_acc_ger = f"evaluation/German/{global_task}_facts/{task}.csv"
    df_rej_eng = pd.read_csv(file_path_rej_eng)
    df_acc_eng = pd.read_csv(file_path_acc_eng)
    df_rej_ger = pd.read_csv(file_path_rej_ger)
    df_acc_ger = pd.read_csv(file_path_acc_ger)
    df_rej_eng.rename(
        columns={"LAR_mean": "lar_score_mean", "LAR_std": "lar_score_std"}, inplace=True
    )
    df_rej_ger.rename(
        columns={"LAR_mean": "lar_score_mean", "LAR_std": "lar_score_std"}, inplace=True
    )
    out_df = compute_lci_score(df_rej_eng, df_acc_eng, df_rej_ger, df_acc_ger)
    print(out_df.shape)
    exit(1)

    lang = "German"
    global_task1 = "nlp_eval"
    task1 = "combined_mini_eval"
    file_path1 = f"evaluation/{lang}/{global_task1}/{task1}.csv"
    df1 = pd.read_csv(file_path1)

    global_task2 = "summary_eval"
    task2 = "summary_eval"
    file_path2 = f"evaluation/{lang}/{global_task2}/{task2}.csv"
    df2 = pd.read_csv(file_path2)
    agg_df2 = aggregate_scores(file_path2, lang)[:30]

    agg_dict = {}
    for col in [
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bert_p",
        "bert_r",
        "bert_f1",
        # "LAR",
    ]:
        df1[f"{col}_mean"] = agg_df2[f"{col}_mean"]
        df1[f"{col}_std"] = agg_df2[f"{col}_std"]

    df1["LAR_mean"] = 0.876
    df1["LAR_std"] = 0.109
    # print(df1.shape)

    # df1.to_csv(f"evaluation/{lang}/{global_task1}/combined_eval.csv", index= False)
