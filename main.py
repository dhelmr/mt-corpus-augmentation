import argparse
import dataclasses
import json
import os
import typing
import enum

import transformers
import spacy
from tqdm import tqdm


class RewritingMode(enum.Enum):
    REPLACE = "replace"
    INSERT = "insert"



SUPPORTED_MASK_MODELS = {
    "bert-base-cased" : "[MASK]",
    "bert-large-cased": "[MASK]",
    "roberta-large": "<mask>",
    "roberta-base": "<mask>",
    "distilbert-base-cased": "[MASK]"
}


class MaskRewriter:
    """
    Rewrites english sentences using a fill-mask hugginggace model. Two modes are supported:
    "replace" replaces tokens of specific pos tags with the mask token, whereas "insert" prepends the mask token
    in front of them. The pos tags for which this is done are defined in the sets "replace_pos_tags",
    resp. "insert_pos_tags".
    The fill-mask model suggests a list of predictions for the mask token which are used for rewriting the sentence
    (each chosen token results in a new sentence). The decision which predicted tokens are chosen can be influenced
    by two parameters:
    - "new_token_score_threshold" is the threshold that the predicted token's score must exceed in order to be chosen
    - only for "replace" mode: "original_token_score_threshold" defines a value which the score of the token that gets
    replaced (i.e. the token in the original sentence) must exceed. The rationale behind this is that the fill-mask
    model should appropriately represent the overall context/meaning of the sentence around the token,
    before trying to replace it. If this value is set to zero, this check is effectively skipped.

    A list of appropiate models is defined by SUPPORTED_MASK_MODELS
    """
    def __init__(self, mode: RewritingMode, model_name):
        if model_name not in SUPPORTED_MASK_MODELS:
            raise ValueError(f"{model_name} is no supported mask model. Choose one of: {SUPPORTED_MASK_MODELS.keys()}")
        self.mode: RewritingMode = mode
        self.pos_tagger = spacy.load("en_core_web_lg")
        self.mask_token = SUPPORTED_MASK_MODELS[model_name]
        self.bert_unmasker = transformers.pipeline(
            "fill-mask", model=model_name
        )
        self.replace_pos_tags = {"ADJ", "ADV", "NOUN", "VERB"}
        self.insert_pos_tags = {"ADJ", "ADV", "NOUN", "VERB"}
        self.ignore_tokens = {"", ";", " ", "\t", ".", ",", "(", ")", "[", "]"}
        self.original_token_score_threshold = 0.01
        self.new_token_score_threshold = 0.01


    def transform(self, sentence: str, context_before: str, context_after: str) -> typing.List[str]:
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
            pred["token_str"].strip()
            for pred in predictions
            if pred["token_str"].lower().strip() != original_token
            and pred["token_str"].lower().strip() not in self.ignore_tokens
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
    """
    dataclass representing a parallel corpus entry, consisting of an english and foreign part
    and additional context for the english part (= the lines before and after)
    """
    english_text: str
    context_before: str
    context_after: str
    foreign_text: str


class CorpusReader:
    """
    Reads a parallel english-foreign corpus line by line
    It provides context information for each english line, i.e. the lines before and after. The context is defined
    by the window_size attribute.
    """
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


class TranslateTransformer:
    """
    Rewrites a sentence using a chain of huggingface transformer models. In principal, any kind of model can
    be specified here. For the task of rewriting an english sentence preserving its meaning, a model chain consisting
    of translating transformers from english to any arbitrary language(s) and back,
    e.g. English -> German -> Russian -> English

    The parameter "n_iterations" defines how many times the model chain is (recursively) applied to each sentence. E.g.,
    a value of 2 means that the original sentence is transformed using the specified model chain, and then the result
    is fed to the model chain again. Each of the interim sentences are part of the output rewritten sentence.
    If a sentence did already occur as a results in an earlier iteration, the rewriting process for a sentence is stopped.
    """
    def __init__(self, model_names: typing.List[str], n_iterations: int):
        self.translators = [
           transformers.pipeline("translation", model_name) for model_name in model_names
        ]
        self.n_iterations = n_iterations

    def _translate_with(self, tokenizer, model, text):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def _transform_with_chain(self, sentence):
        representation = sentence
        for model in self.translators:
            representation = model(representation)[0]["translation_text"]
        return representation

    def transform(self, sentence: str, context_before="", context_after="") -> typing.List[str]:
        transformed_sentences = set()
        current_input = sentence
        for i in range(self.n_iterations):
            transformed_english = self._transform_with_chain(current_input)
            if transformed_english.lower() == sentence.lower() or transformed_english in transformed_sentences:
                break
            transformed_sentences.add(transformed_english)
            current_input = transformed_english
        return list(transformed_sentences)


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
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output"
    )
    parser.add_argument(
        "-w",
        "--window",
        help="Number of sentences before and after each selected sentence to pass to the transformer model to give it more context.",
        type=int,
        default=1,
    )
    subparsers = parser.add_subparsers(dest="mode")
    masked_parser = subparsers.add_parser("fill-mask")
    masked_parser.add_argument(
        "--new-token-score-threshold",
        help="Only predicted tokens with a score above [factor*original_score] will be taken.",
        default=0.2,
        type=float,
    )
    masked_parser.add_argument(
        "--original-token-score-threshold",
        help="Only tokens whose prediction score is above this threshold will be replaced.",
        type=float,
        default=0,
    )
    masked_parser.add_argument(
        "--model",
        help=f"Name of the huggingface fill-mask model. Choose one of: {SUPPORTED_MASK_MODELS.keys()}",
        type=str,
        default="distilbert-base-cased"
    )
    masked_parser.add_argument(
        "--fill-mask-mode",
        help=f"Mode how the generated token will be used for rewriting. Choose of: {[v.value for v in RewritingMode]}",
        default=RewritingMode.REPLACE,
        type=RewritingMode
    )
    transform_parser = subparsers.add_parser("translate-transform")
    transform_parser.add_argument(
        "--model-chain", help="Model chain used for transforming the sentence step by step",
        nargs="+",
        default=["facebook/wmt19-en-de","facebook/wmt19-de-en"]
    )
    transform_parser.add_argument(
        "--iter", help="Number of translation iterations per sentence",
        default=3,
        type=int
    )
    parsed = parser.parse_args()

    if parsed.mode == "fill-mask":
        transformer = MaskRewriter(parsed.fill_mask_mode, parsed.model)
        transformer.original_token_score_threshold = parsed.original_token_score_threshold
        transformer.new_token_score_threshold = parsed.new_token_score_threshold
    elif parsed.mode == "translate-transform":
        transformer = TranslateTransformer(model_names=parsed.model_chain,
                                           n_iterations=parsed.iter)
        if parsed.window > 0:
            print("Note that the '--window' option does not have any effect for this mode.")
    else:
        raise ValueError(f"Unknown mode: {parsed.mode}")
    reader = CorpusReader(english_path=parsed.english, foreign_path=parsed.foreign, window_size=parsed.window)
    corpus_writer = ParallelCorpusWriter(dir_path=parsed.output)
    iterator = reader.read()
    if not parsed.verbose:
        iterator = tqdm(iterator, total=reader.number_of_lines())
    # write configuration for later reference
    config_path = os.path.join(parsed.output, "config.json")
    config_json = parsed.__dict__.copy()
    if "fill_mask_mode" in config_json:
        config_json["fill_mask_mode"] = config_json["fill_mask_mode"].value
    with open(config_path, "w") as f:
        json.dump(config_json, f)
    # start reading sentence by sentence
    for sentence in iterator:
        new_sentences = transformer.transform(
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
