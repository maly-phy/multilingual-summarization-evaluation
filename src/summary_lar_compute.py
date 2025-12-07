import pandas as pd


class SummaryLAR:
    def __init__(self, df_path, language):
        self.df_path = df_path
        self.language = language

    def compute_lar(self, ref_summary, candidate_summary):
        lar_metric = 1 - abs(len(ref_summary) - len(candidate_summary)) / len(
            ref_summary
        )
        return max(0.0, lar_metric)

    def process_lar(self):
        df = pd.read_csv(self.df_path)
        all_results = []
        for idx, row in df.iterrows():
            ref_summary = row["ref_summary"]
            candidate_summary = row["model_summary"]
            lar_metric = self.compute_lar(ref_summary, candidate_summary)
            all_results.append({"LAR": lar_metric})

            if idx % 5 == 0:
                print(f"Processing {idx} / {len(df)}\n")

        df = pd.concat([df, pd.DataFrame(all_results)], axis=1)
        df.to_csv(self.df_path, index=False)
        return df


if __name__ == "__main__":
    language = "English"
    df_path = (
        f"evaluation/{language}/summary_eval/llama-3.1-8b-instant_summary_eval.csv"
    )
    summary_lar = SummaryLAR(df_path, language)
    output_df = summary_lar.process_lar()
