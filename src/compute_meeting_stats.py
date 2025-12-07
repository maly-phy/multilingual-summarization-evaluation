import nltk
from collections import Counter
import ast, re
import pandas as pd
from utils import merge_data_files
import os

# nltk.download("punkt")
# nltk.download("punkt_tab")


class SyntheticMeetingAnalyzer:
    def __init__(self, input_df, language):
        self.input_df = input_df
        self.language = language
        self.meeting_data = self.load_meetings()

    def load_meetings(self):
        def extract_speaker_and_dialog(line):
            pattern = r"^(.*?)(?:\s*\([^)]*\))?\s*:\s*(.*)"
            match = re.match(pattern, line)
            if match:
                speaker = match.group(1).strip()
                dialog = match.group(2).strip()
                return speaker, dialog
            else:
                return None, None

        meetings = []
        required_columns = [
            "Title",
            "Article",
            "Tags",
            "Personas",
            "Summary",
            "Meeting_Plan",
            "Meeting",
        ]
        for idx, row in self.input_df.iterrows():
            if pd.isna(row["Article"]) or row["Article"] == "":
                print(f"Article column is empty or NaN in row {idx}.")
                continue
            meeting_plan = (
                ast.literal_eval(row["Meeting_Plan"])
                if pd.notna(row["Meeting_Plan"])
                else []
            )
            tags = ast.literal_eval(row["Tags"]) if pd.notna(row["Tags"]) else []
            personas = (
                ast.literal_eval(row["Personas"]) if pd.notna(row["Personas"]) else []
            )
            participants = [p["role"] for p in personas if "role" in p]

            turns = row["Meeting"].split(">>")
            turns = [turn.strip() for turn in turns if turn.strip()]
            turns_dict = []
            for turn in turns:
                if turn != "":
                    if ":" in turn:
                        speaker, dialog = extract_speaker_and_dialog(turn)
                        if speaker and dialog:
                            turns_dict.append(
                                {"speaker": speaker.strip(), "dialog": dialog.strip()}
                            )
                    else:
                        turns_dict.append(
                            {"speaker": "Unknown", "dialog": turn.strip()}
                        )

            meetings.append(
                {
                    "title": row["Title"],
                    "article": row["Article"],
                    "tags": tags,
                    "personas": personas,
                    "participants": participants,
                    "summary": row["Summary"],
                    "meeting_plan": meeting_plan,
                    "turns": turns_dict,
                    "meeting_type": row["Meeting_Type"],
                }
            )
        return meetings

    def compute_general_stats(self):
        num_meetings = len(self.meeting_data)
        type_counts = Counter([m["meeting_type"] for m in self.meeting_data])
        avg_participants = (
            sum([len(m["participants"]) for m in self.meeting_data]) / num_meetings
            if num_meetings > 0
            else 0
        )
        general_stats = {
            "num_meetings": num_meetings,
            "meeting_per_type": dict(type_counts),
            "avg_num_participants": round(avg_participants, 3),
        }
        return general_stats

    def compute_turn_level_stats(self):
        num_meetings = len(self.meeting_data)
        total_turns = sum(len(m["turns"]) for m in self.meeting_data)
        total_words = sum(
            len(self.tokenize(turn["dialog"]))
            for m in self.meeting_data
            for turn in m["turns"]
        )
        total_scenes = sum(len(m["meeting_plan"]) for m in self.meeting_data)
        avg_turns_per_meeting = total_turns / num_meetings if num_meetings > 0 else 0
        avg_words_per_meeting = total_words / num_meetings if num_meetings > 0 else 0
        avg_words_per_turn = total_words / total_turns if total_turns > 0 else 0
        avg_scenes_per_meeting = total_scenes / num_meetings if num_meetings > 0 else 0

        turn_level_stats = {
            "total_turns": total_turns,
            "total_words": total_words,
            "total_scenes": total_scenes,
            "avg_turns_per_meeting": round(avg_turns_per_meeting, 3),
            "avg_words_per_meeting": round(avg_words_per_meeting, 3),
            "avg_words_per_turn": round(avg_words_per_turn, 3),
            "avg_scenes_per_meeting": round(avg_scenes_per_meeting, 3),
        }
        return turn_level_stats

    def compute_linguistic_diversity(self):
        all_tokens = []
        for meeting in self.meeting_data:
            for turn in meeting["turns"]:
                tokens = self.tokenize(turn["dialog"])
                all_tokens.extend(tokens)

        vocabs = set(all_tokens)
        total_tokens = len(all_tokens)
        type_token_ratio = len(vocabs) / total_tokens if total_tokens else 0
        ling_div_stats = {
            "vocab_size": len(vocabs),
            "total_tokens": total_tokens,
            "type_token_ratio": round(type_token_ratio, 5),
        }
        return ling_div_stats

    def compute_summary_stats(self):
        num_meetings = len(self.meeting_data)
        total_summary_words = sum(
            len(self.tokenize(m["summary"])) for m in self.meeting_data
        )
        avg_summary_len_words = (
            total_summary_words / num_meetings if num_meetings else 0
        )
        summary_stats = {
            "total_summary_words": total_summary_words,
            "avg_summary_length_in_words": round(avg_summary_len_words, 3),
        }
        return summary_stats

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text.lower(), language=self.language.lower())
        tokens_wout_puncts = [t for t in tokens if t.isalnum()]
        return tokens_wout_puncts

    def compute_ngram_overlap(self, text_a, text_b, n=2):
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b)
        if len(tokens_a) < n or len(tokens_b) < n:
            return 0.0

        # ngrams_a = set(self.get_ngrams(tokens_a, n))
        # ngrams_b = set(self.get_ngrams(tokens_b, n))
        ngrams_a = set(nltk.ngrams(tokens_a, n))
        ngrams_b = set(nltk.ngrams(tokens_b, n))
        overlap = ngrams_a.intersection(ngrams_b)
        overlap_ratio = len(overlap) / max(len(ngrams_a), len(ngrams_b))
        return round(overlap_ratio, 5)

    def get_ngrams(self, tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def evaluate_all(self):
        all_stats = {
            "general": self.compute_general_stats(),
            "turn_level": self.compute_turn_level_stats(),
            "linguistic_diversity": self.compute_linguistic_diversity(),
            "summary": self.compute_summary_stats(),
        }
        return all_stats

    def write_state_to_file(self, filename):
        results = self.evaluate_all()
        with open(filename, "w") as f:
            for stat_category, stats in results.items():
                f.write(f"\n=== {stat_category.upper()} ===\n")
                for metric, value in stats.items():
                    f.write(f"{metric}: {value}\n")

        print(f"Meeting analysis written to {filename}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    language = "German"
    eng_meetings_df = merge_data_files("data/fame_dataset", f"{language}")
    analyzer = SyntheticMeetingAnalyzer(eng_meetings_df, language=language)
    analyzer.write_state_to_file(
        f"outputs/statistic_analysis/meeting_analysis_{language.lower()}.txt"
    )
