import nltk
import re, string
from utils import merge_data_files
import pandas as pd

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 500)
pd.set_option("display.max_colwidth", 500)


class Cleaner:
    def __init__(self, df, language):
        self.df = df
        self.language = language

    def remove_double_spaces(self, text):
        return re.sub(" +", " ", text)

    def lowercase(self, text):
        return text.lower()

    def remove_blank_lines(self, text):
        return re.sub("\n+", "\n", text).strip()

    def remove_interpunctuation(self, text):
        all_tokens = nltk.word_tokenize(text, language=self.language.lower())
        tokens_wout_puncts = [
            tok for tok in all_tokens if tok not in string.punctuation
        ]
        return tokens_wout_puncts

    def preprocess_corpus(self):
        columns = ["Article", "Summary", "Meeting"]
        for col in columns:
            self.df[col] = self.df[col].apply(self.lowercase)
            self.df[col] = self.df[col].apply(self.remove_blank_lines)
            self.df[col] = self.df[col].apply(self.remove_double_spaces)

        return self.df


if __name__ == "__main__":
    language = "German"
    file_path = "data/fame_dataset"
    df = merge_data_files(file_path, language)
    cleaner = Cleaner(df, language)
    cleaned_df = cleaner.preprocess_corpus()
    columns = ["Article", "Summary", "Meeting"]
    print(cleaned_df[columns].head(2))
