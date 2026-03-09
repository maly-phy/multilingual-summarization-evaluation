import pandas as pd
import os
import ast


class SummQualityScorer:
    def __init__(self, language, exclude_criteria=None):
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
        self.exclude_criteria = exclude_criteria

    def impact_scores(self, row, criteria, weight):
        criterion = ast.literal_eval(row[criteria] if criteria else row)
        impact_score = float(criterion["impact_score"])
        confidence_score = float(criterion["confidence_score"]) / 10
        numerator = impact_score * confidence_score * weight
        denominator = confidence_score * weight
        return numerator, denominator

    def calculate_overall_quality(self, nums, denos):
        overall_impact = nums / denos
        summary_quality = 1 + (((5 - overall_impact) / 5) * 9)
        return overall_impact, summary_quality

    def weighted_severity_impact(self, row):
        nums, denos = 0.0, 0.0
        for criteria, weight in self.weight_importances.items():
            if self.exclude_criteria and criteria in self.exclude_criteria:
                continue
            numerator, denominator = self.impact_scores(row, criteria, weight)
            nums += numerator
            denos += denominator

        return self.calculate_overall_quality(nums, denos)

    def process_summary_quality(self, severity_impact_df):
        for idx, row in severity_impact_df.iterrows():
            overall_impact, summary_quality = self.weighted_severity_impact(row)
            severity_impact_df.at[idx, "weighted_impact_score"] = overall_impact
            severity_impact_df.at[idx, "summary_quality_score"] = summary_quality

        save_dir = f"multiagent_summary/evaluation/{self.language}/error_based/summary_quality.csv"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        severity_impact_df.to_csv(save_dir, index=False)
        print(f"Quality results saved to {save_dir}")


if __name__ == "__main__":
    language = "English"
    severity_impact_path = (
        f"multiagent_summary/evaluation/{language}/error_based/severity_impact.csv"
    )
    severity_impact_df = pd.read_csv(severity_impact_path)
    scorer = SummQualityScorer(language, exclude_criteria=None)
    scorer.process_summary_quality(severity_impact_df)
    out_df = pd.read_csv(
        f"multiagent_summary/evaluation/{language}/error_based/summary_quality.csv"
    )
    print(out_df["summary_quality_score"].min())
