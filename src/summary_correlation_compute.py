from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
import os


class HumanModelCorrelation:
    def __init__(self, human_scores_path, model_scores_path, save_path):
        self.human_scores_path = human_scores_path
        self.model_scores_path = model_scores_path
        self.save_path = save_path

    def spearman_corr(self, x, y):
        corr, p_value = spearmanr(x, y)
        return corr, p_value

    def pearson_corr(self, x, y):
        corr, p_value = pearsonr(x, y)
        return corr, p_value

    def kendall_corr(self, x, y):
        corr, p_value = kendalltau(x, y)
        return corr, p_value

    def compute_correlations(self, human_col, model_col, save_path):
        human_df = pd.read_csv(self.human_scores_path)
        model_df = pd.read_csv(self.model_scores_path)  # for German take model_df[:28]
        human_scores = human_df[human_col].tolist()
        model_scores = model_df[model_col].tolist()

        spearman_corr, spearman_p = self.spearman_corr(human_scores, model_scores)
        pearson_corr, pearson_p = self.pearson_corr(human_scores, model_scores)
        kendall_corr, kendall_p = self.kendall_corr(human_scores, model_scores)

        results = {
            "spearman_corr": round(spearman_corr, 3),
            "spearman_p": round(spearman_p, 3),
            "pearson_corr": round(pearson_corr, 3),
            "pearson_p": round(pearson_p, 3),
            "kendall_corr": round(kendall_corr, 3),
            "kendall_p": round(kendall_p, 3),
        }
        output_df = pd.DataFrame([results])
        output_df.to_csv(save_path, index=False)

        return output_df


if __name__ == "__main__":
    language = "German"
    task = "summary_eval"
    human_scores_path = f"evaluation/{language}/{task}/{task}_human.csv"
    model_scores_path = f"evaluation/{language}/{task}/llama-3.1-8b-instant_{task}_with_naturalness_0_29.csv"
    save_path = f"evaluation/{language}/{task}/{task}_correlation.csv"
    correlator = HumanModelCorrelation(
        human_scores_path,
        model_scores_path,
        save_path,
    )
    output_df = correlator.compute_correlations(
        "naturalness_score",
        "Naturalness_base_score",
        save_path,
    )
    print(output_df)
