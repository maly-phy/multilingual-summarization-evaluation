import warnings

warnings.filterwarnings("ignore")
import evaluate
import nltk
from nltk.translate import meteor
from nltk import word_tokenize
import sys, os

import torch
import pandas as pd
from bleurt import score

sys.path.append(os.getcwd())
from submodules.QuestEval.questeval.questeval_metric import QuestEval
from submodules.LongDocFACTScore.src.longdocfactscore.ldfacts import (
    BARTScore,
    LongDocFACTScore,
)
from submodules.blanc.blanc.blanc import BlancHelp, BlancTune
from submodules.LENS.lens.lens.lens_score import LENS
from submodules.LENS.lens.lens.models import download_model

# from submodules.moverscore.moverscore_v2 import get_idf_dict, word_mover_score
# nltk.download("punkt")
# nltk.download("wordnet")
bleurt_ckpt = "BLEURT-20-D12"


class NLPMetricEvaluator:
    def __init__(self, df, language, device, metric_type, save_path):
        self.df = df
        self.language = language
        self.device = device
        self.metric_type = metric_type
        self.save_path = save_path
        self.ldfacts = LongDocFACTScore(device=self.device, language=self.language)
        self.bart = BARTScore(device=self.device)
        self.bleurt = score.BleurtScorer(bleurt_ckpt)
        self.questeval = QuestEval(no_cuda=False, language="en")
        self.blanc_help = BlancHelp(
            device=self.device, inference_batch_size=128, show_progress_bar=False
        )
        self.blanc_tune = BlancTune(
            device=self.device,
            inference_batch_size=24,
            finetune_mask_evenly=False,
            finetune_batch_size=24,
            show_progress_bar=False,
        )
        self.lens_path = download_model("davidheineman/lens")
        self.lens = LENS(self.lens_path, rescale=True)

    def compute_questeval(self, src, pred, ref):
        results = self.questeval.corpus_questeval(
            hypothesis=pred, sources=src, list_references=[ref]
        )["ex_level_scores"]
        return list(results)

    def compute_meteor(self, ref, pred):
        scores = []
        for g, p in zip(ref, pred):
            a = [word_tokenize(g)]
            b = word_tokenize(p)
            scores.append(meteor(a, b))
        return scores

    def compute_bleu(self, ref, pred):
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(predictions=pred, references=ref)
        return bleu_score["bleu"]

    def compute_bleurt(self, ref, pred):
        bleurt_score = self.bleurt.score(references=ref, candidates=pred, batch_size=4)
        return bleurt_score

    def compute_chrf(self, ref, pred):
        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(predictions=pred, references=ref)
        return chrf_score["score"]

    def compute_perplexity(self, pred):
        perplexity = evaluate.load("perplexity")
        perplexity_score = perplexity.compute(predictions=pred, model_id="gpt2")
        return perplexity_score["perplexities"]

    def compute_bart(self, ref, pred):
        bart_metric = self.bart.bart_score(ref, pred)
        return bart_metric

    def compute_ldfacts(self, src, pred):
        ldfactscore = self.ldfacts.score_src_hyp_long(src, pred)
        return ldfactscore

    def compute_blanc(self, src, pred):
        blanc_help_score = self.blanc_help.eval_once(src, pred)
        blanc_tune_score = self.blanc_tune.eval_once(src, pred)
        return blanc_help_score, blanc_tune_score

    def compute_lens(self, src, ref, pred):
        lens_score = self.lens.score(
            src,
            pred,
            [ref],
            batch_size=8,
            devices=[0] if torch.cuda.is_available() else None,
        )
        return lens_score[0]

    def process_nlp_evaluation(self):
        df = self.df[:30]
        start_idx = df.index[0]
        end_idx = df.index[-1]
        results = []
        for idx, row in df.iterrows():
            src = [row["Meeting"]]
            pred = [row["model_summary"]]
            ref = [row["ref_summary"]]

            if idx % 5 == 0:
                print(f"Processing {idx} / {len(df)}\n")

            if metric_type == "bleurt":
                bleurt_score = self.compute_bleurt(ref, pred)
                results.append(
                    {
                        "bleurt_score": round(bleurt_score[0], 3),
                    }
                )

            elif self.metric_type == "hf":
                meteor_score = self.compute_meteor(ref, pred)
                bleu_score = self.compute_bleu(ref, pred)
                chrf_score = self.compute_chrf(ref, pred)
                perplexity_score = self.compute_perplexity(pred)
                results.append(
                    {
                        "model_summary": row["model_summary"],
                        "ref_summary": row["ref_summary"],
                        "meteor_score": round(meteor_score[0], 3),
                        "bleu_score": round(bleu_score, 3),
                        "chrf_score": round(chrf_score, 3),
                        "perplexity_score": round(perplexity_score[0], 3),
                    }
                )

            elif self.metric_type == "bart_ldfacts":
                bart_score = self.compute_bart(ref, pred)
                ldfact_score = self.compute_ldfacts(src, pred)
                results.append(
                    {
                        "Meeting": row["Meeting"],
                        "bart_score": round(bart_score[0], 3),
                        "ldfact_score": round(ldfact_score[0], 3),
                    }
                )

            elif self.metric_type == "questeval":
                questeval_score = self.compute_questeval(src, pred, ref)
                results.append(
                    {
                        "questeval_score": round(questeval_score[0], 3),
                    }
                )
            elif self.metric_type == "blanc":
                blank_help_score, blank_tune_score = self.compute_blanc(src[0], pred[0])
                results.append(
                    {
                        "blanc_help_score": round(blank_help_score, 3),
                        "blanc_tune_score": round(blank_tune_score, 3),
                    }
                )

            elif self.metric_type == "lens":
                lens_score = self.compute_lens(src, ref, pred)
                results.append(
                    {
                        "lens_score": round(lens_score, 3),
                    }
                )

        output_df = pd.DataFrame(results)
        self.save_path = self.save_path.replace(".csv", f"_{start_idx}_{end_idx}.csv")
        output_df.to_csv(self.save_path, index=False)
        print(f"Saved results to {self.save_path}")

        return output_df


if __name__ == "__main__":
    language = "English"
    task = "nlp_eval"
    device = torch.device("cuda")
    metric_type = "lens"
    save_dir = f"evaluation/{language}/{task}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{metric_type}_eval.csv"
    input_path = (
        f"evaluation/{language}/summary_eval/llama-3.1-8b-instant_summary_eval.csv"
    )
    df = pd.read_csv(input_path, encoding="utf-8")
    nlp_evalator = NLPMetricEvaluator(df, language, device, metric_type, save_path)
    output_df = nlp_evalator.process_nlp_evaluation()
