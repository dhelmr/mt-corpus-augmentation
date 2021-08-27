import argparse
import dataclasses
import os
import typing
import enum

import transformers
import spacy
from tqdm import tqdm


class RewritingMode(enum.Enum):
    REPLACE = "replace"
    INSERT = "insert"


SUPPORTED_MODELS = {
    "bert-base-cased" : "[MASK]",
    "bert-large-cased": "[MASK]",
    "roberta-large": "<mask>",
    "roberta-base": "<mask>",
    "distilbert-base-cased": "[MASK]"
}


class Rewriter:
    def __init__(self, mode: RewritingMode, model_name):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"{model_name} is no supported model. Choose one of: {SUPPORTED_MODELS.keys()}")
        self.mode: RewritingMode = mode
        self.pos_tagger = spacy.load("en_core_web_lg")
        self.mask_token = SUPPORTED_MODELS[model_name]
        self.bert_unmasker = transformers.pipeline(
            "fill-mask", model=model_name
        )
        self.replace_pos_tags = {"ADJ", "ADV", "NOUN", "VERB"}
        self.insert_pos_tags = {"ADJ", "ADV", "NOUN", "VERB"}
        self.original_token_score_threshold = 0.01
        self.new_token_score_threshold = 0.01


    def rewrite(self, sentence: str, context_before: str, context_after: str):
        rewritten_sentences = []

        tokens = self.pos_tagger(sentence)
        for i, tok in enumerate(tokens):
            if self.mode == RewritingMode.REPLACE and tok.pos_ not in self.replace_pos_tags:
                continue
            elif self.mode == RewritingMode.INSERT and tok.pos_ not in self.insert_pos_tags:
                continue
            new_tokens = self._best_tokens(
                tokens, i, context_before, context_after
            )
            for new_token in new_tokens:
                new_sentence = self._make_new_sentence(tokens, i, new_token)
                rewritten_sentences.append(new_sentence)
        return rewritten_sentences

    def _best_tokens(
        self, tokens, i: int, prefix: str, suffix: str
    ) -> typing.List[str]:
        original_token = tokens[i].text.lower()
        masked_sentence = self._make_new_sentence(tokens, i, self.mask_token)
        with_context = prefix + masked_sentence + suffix
        predictions = self.bert_unmasker(with_context)
        original_token_score = 0
        for pred in predictions:
            if pred["token_str"].lower() == original_token:
                original_token_score = pred["score"]
                break
        if original_token_score < self.original_token_score_threshold:
            return []
        best_replacements = [
            pred["token_str"]
            for pred in predictions
            if pred["token_str"].lower() != original_token
            and pred["score"] >= self.new_token_score_threshold
        ]
        return best_replacements

    def _make_new_sentence(self, tokens, index: int, new_token: str):
        if self.mode == RewritingMode.REPLACE:
            new_tokens = [
                tok.text if i != index else new_token
                for i, tok in enumerate(tokens)
            ]
        elif self.mode == RewritingMode.INSERT:
            new_tokens = [tok.text for tok in tokens]
            new_tokens.insert(index, new_token)
        else:
            raise RuntimeError("Unknown mode")
        return " ".join(new_tokens)


@dataclasses.dataclass
class CorpusSentence:
    english_text: str
    context_before: str
    context_after: str
    foreign_text: str


class CorpusReader:
    def __init__(self, english_path: str, foreign_path: str, window_size: int = 5):
        self.window_size = window_size
        self.english_path = english_path
        self.foreign_path = foreign_path

    def read(self) -> typing.Iterable[CorpusSentence]:
        context_before = [""] * self.window_size
        context_after = []
        with open(self.english_path, "r") as e, open(self.foreign_path) as f:
            for i in range(self.window_size):
                context_after.append(e.readline().strip())
            for next_line in e:
                next_line = next_line.strip()
                sentence = context_after.pop(0)
                context_after.append(next_line)
                foreign_sentence = f.readline().strip()
                yield CorpusSentence(
                    sentence,
                    "".join(context_before),
                    "".join(context_after),
                    foreign_sentence,
                )
                context_before.pop(0)
                context_before.append(sentence)

    def number_of_lines(self) -> int:
        with open(self.english_path) as f:
            count = sum(1 for _ in f)
        return count


class ParallelCorpusWriter:
    def __init__(self, dir_path: str):
        self.foreign_path = os.path.join(dir_path, "foreign.txt")
        self.english_path = os.path.join(dir_path, "english.txt")

    def write_sentence_pair(self, english: str, foreign: str):
        with open(self.foreign_path, "a") as f, open(self.english_path, "a") as e:
            e.write(english + "\n")
            f.write(foreign + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--foreign",
        help="Foreign Input Corpus (won't be rewritten)",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--english",
        help="English Input Corpus (sentences will be rewritten)",
        required=True,
    )
    parser.add_argument("-o", "--output", help="Output directory path", default=".")
    parser.add_argument(
        "--new-token-score-threshold",
        help="Only predicted tokens with a score above [factor*original_score] will be taken.",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--original-token-score-threshold",
        help="Only tokens whose prediction score is above this threshold will be replaced.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-w",
        "--window-size",
        help="Number of sentences before and after each selected sentence to pass to the unmasker to give it more context.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model",
        help=f"Name of the huggingface fill-mask model. Choose one of: {SUPPORTED_MODELS.keys()}",
        type=str,
        default="distilbert-base-cased"
    )
    parser.add_argument(
        "--mode",
        help=f"Mode how the generated token will be used for rewriting. Choose of: {[v.value for v in RewritingMode]}",
        default=RewritingMode.REPLACE,
        type=RewritingMode
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output"
    )
    parsed = parser.parse_args()

    rewriter = Rewriter(parsed.mode, parsed.model)
    rewriter.original_token_score_threshold = parsed.original_token_score_threshold
    rewriter.new_token_score_threshold = parsed.new_token_score_threshold
    reader = CorpusReader(english_path=parsed.english, foreign_path=parsed.foreign, window_size=parsed.window_size)
    corpus_writer = ParallelCorpusWriter(dir_path=parsed.output)
    iterator = reader.read()
    if not parsed.verbose:
        iterator = tqdm(iterator, total=reader.number_of_lines())
    for sentence in iterator:
        new_sentences = rewriter.rewrite(
            sentence.english_text, sentence.context_before, sentence.context_after
        )
        if parsed.verbose:
            print("Original English: ", sentence.english_text)
            print("Original Foreign: ", sentence.foreign_text)
            print(f"Generated Sentences ({len(new_sentences)}):")

            for ns in new_sentences:
                print("\t", ns)
            print("------------------------------------------\n")
        for ns in new_sentences:
            corpus_writer.write_sentence_pair(foreign=sentence.foreign_text, english=ns)


if __name__ == "__main__":
    main()
