import argparse
import dataclasses
import os
import pprint
import typing

import transformers
import spacy
from tqdm import tqdm


class Rewriter:
    def __init__(self):
        self.pos_tagger = spacy.load("en_core_web_lg")
        self.bert_unmasker = transformers.pipeline(
            "fill-mask", model="bert-base-uncased"
        )
        self.skip_pos_tokens = {".", ":", "``", ",", "''", "-LRB-", "-RRB-", "HYPH"}
        self.original_token_score_threshold = 0.01
        self.replacement_score_factor = 1

    def rewrite(self, sentence: str, context_before: str, context_after: str):
        rewritten_sentences = []

        pos_tokens = self.pos_tagger(sentence)
        for i, tok in enumerate(pos_tokens):
            if tok.tag_ in self.skip_pos_tokens:
                continue
            replacements = self._best_replacements(
                pos_tokens, i, context_before, context_after
            )
            for replacement in replacements:
                new_sentence = self._make_new_sentence(pos_tokens, i, replacement)
                rewritten_sentences.append(new_sentence)
        return rewritten_sentences

    def _best_replacements(
        self, tokens, i: int, prefix: str, suffix: str
    ) -> typing.List[str]:
        original_token = tokens[i].text.lower()
        masked_sentence = self._make_new_sentence(tokens, i, "[MASK]")
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
            and pred["score"] >= original_token_score * self.replacement_score_factor
        ]
        return best_replacements

    def _make_new_sentence(self, tokens, replacement_i: int, new_token: str):
        new_tokens = [
            tok.text if i != replacement_i else new_token
            for i, tok in enumerate(tokens)
        ]
        return " ".join(new_tokens)


@dataclasses.dataclass
class CorpusSentence:
    english_text: str
    context_before: str
    context_after: str
    foreign_text: str


class CorpusReader:
    def __init__(self, context_size: int = 5):
        self.context_size = context_size

    def read(
        self, english_path: str, foreign_path: str
    ) -> typing.Iterable[CorpusSentence]:
        context_before = [""] * self.context_size
        context_after = []
        with open(english_path, "r") as e, open(foreign_path) as f:
            for i in range(self.context_size):
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


class ParallelCorpusWriter:
    def __init__(self, dir_path: str):
        self.foreign_path = os.path.join(dir_path, "foreign.txt")
        self.english_path = os.path.join(dir_path, "english.txt")

    def write_sentence_pair(self, english: str, foreign: str):
        with open(self.foreign_path, "a") as f, open(self.english_path, "a") as e:
            e.write(english+"\n")
            f.write(foreign+"\n")


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
        "--replacement-score-factor",
        help="Only predicted tokens with a score above [factor*original_score] will be taken.",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--original-token-threshold",
        help="Only tokens whose prediction score is above this threshold will be replaced.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-c",
        "--context-size",
        help="Number of sentences before and after each selected sentence to pass to the unmasker to give it more context.",
        type=int,
        default=3,
    )
    parsed = parser.parse_args()

    rewriter = Rewriter()
    rewriter.original_token_score_threshold = parsed.original_token_threshold
    rewriter.replacement_score_factor = parsed.replacement_score_factor
    reader = CorpusReader(context_size=parsed.context_size)
    corpus_writer = ParallelCorpusWriter(dir_path=parsed.output)
    for sentence in reader.read(
        english_path=parsed.english, foreign_path=parsed.foreign
    ):
        new_sentences = rewriter.rewrite(
            sentence.english_text, sentence.context_before, sentence.context_after
        )
        print("Original English: ", sentence.english_text)
        print("Original Foreign: ", sentence.foreign_text)
        print(f"Generated Sentences ({len(new_sentences)}):")

        for ns in new_sentences:
            print("\t", ns)
            corpus_writer.write_sentence_pair(foreign=sentence.foreign_text, english=ns)
        print("------------------------------------------\n")


if __name__ == "__main__":
    main()
