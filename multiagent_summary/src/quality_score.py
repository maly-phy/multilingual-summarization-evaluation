import pandas as pd
import os
import ast


class SummQualityScorer:
    def __init__(self, severity_impact_df, language):
        self.severity_impact_df = severity_impact_df
        self.language = language
        self.weight_importances = {
            "Hallucination": 1.0,
            "Omission": 1.1,
            "Irrelevance": 1.1,
            "Redundancy": 0.9,
            "Incoherence": 0.9,
            "Linguistic Inaccuracy": 0.9,
            "Structure": 0.9,
        }

    def weighted_severity_impact(self, idx):
        nums, denos = [], []
        for criteria, weight in self.weight_importances.items():
            criterion = ast.literal_eval(self.severity_impact_df.at[idx, criteria])
            impact_score = float(criterion["impact_score"])
            confidence_score = float(criterion["confidence_score"]) / 10
            numerator = impact_score * confidence_score * weight
            denominator = confidence_score * weight
            nums.append(numerator)
            denos.append(denominator)

        overall_impact = sum(nums) / sum(denos) if sum(denos) != 0 else 0
        summary_quality = 1 + (((5 - overall_impact) / 5) * 9)
        return overall_impact, summary_quality

    def process_summary_quality(self):
        for idx, row in self.severity_impact_df.iterrows():
            overall_impact, summary_quality = self.weighted_severity_impact(idx)
            self.severity_impact_df.at[idx, "weighted_impact_score"] = overall_impact
            self.severity_impact_df.at[idx, "summary_quality_score"] = summary_quality

        save_dir = f"multiagent_summary/evaluation/{self.language}/error_based/summary_quality.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        self.severity_impact_df.to_csv(save_dir, index=False)
        print(f"Quality results saved to {save_dir}")


if __name__ == "__main__":
    language = "English"
    severity_impact_path = (
        f"multiagent_summary/evaluation/{language}/error_based/severity_impact.csv"
    )
    severity_impact_df = pd.read_csv(severity_impact_path)
    # scorer = SummQualityScorer(severity_impact_df, language)
    # scorer.process_summary_quality()
    out_df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/error_based/summary_quality.csv"
    )
    print(out_df["summary_quality_score"].min())
