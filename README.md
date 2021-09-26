# Parallel Corpus Augmentation for Machine Translation

This repository contains a python script that uses huggingface for augmenting 
the english side of a parallel corpus.


# Installation

This needs python >= 3.8. Requirements must be installed with `pip install -r requirements.txt`. 

## Usage
General usage instructions can be printed with:

```
> python main.py --help 
usage: main.py [-h] -f FOREIGN -e ENGLISH [-o OUTPUT] [-v] [-w WINDOW]
               {fill-mask,translate-transform} ...

positional arguments:
  {fill-mask,translate-transform}

optional arguments:
  -h, --help            show this help message and exit
  -f FOREIGN, --foreign FOREIGN
                        Foreign Input Corpus (won't be rewritten)
  -e ENGLISH, --english ENGLISH
                        English Input Corpus (sentences will be rewritten)
  -o OUTPUT, --output OUTPUT
                        Output directory path
  -v, --verbose         Verbose output
  -w WINDOW, --window WINDOW
                        Number of sentences before and after each selected
                        sentence to pass to the transformer model to give it
                        more context.

```


### Rewriting using fill-mask models

The first method uses models of [huggingface's "fill-mask" task](https://huggingface.co/transformers/task_summary.html#masked-language-modeling). Given a sentence containing a predefined mask token, it predicts the best replacements for this token. E.g., in the sentence `Paris is the [MASK] of France.`, the mask token `[MASK]` might best be replaced by `capital`. 

The fill-mask models are used in two ways for augmenting the english side of a parallel corpus: In the first mode, certain part-of-speech tags in the input sentence are replaced by the mask token ("replace" mode), whereas in the second mode, it is preprended in front of them ("insert" mode). For determining the part of speech tags, the python library `spacy` is used. By default, the PoS tags that are taken into consideration are adjectives, adverbs, nouns and verbs.

The fill-mask model then predicts the best substitutions for the mask token. Each such new token is associated to a score which reflects the model's confidence in the word belonging to the given position. A configurable threshold ensures that only tokens above a certain score are used. In the "replace" mode furthermore, it is examined whether the original token (i.e. the token of the original sentence that the mask token replaces) is predicted by the fill-mask model and if so, if its assigned score is above a certain threshold. The rationale behind this it to ensure that the model gets the overall meaning of the sentence at the token context.

A complete explanation of all avaialable parameters for this mode can be displayed as follows:

```
> python main.py fill-mask --help 
usage: main.py fill-mask [-h]
                         [--new-token-score-threshold
                          NEW_TOKEN_SCORE_THRESHOLD]
                         [--original-token-score-threshold 
                         ORIGINAL_TOKEN_SCORE_THRESHOLD]
                         [--model MODEL] [--fill-mask-mode FILL_MASK_MODE]

optional arguments:
  -h, --help            show this help message and exit
  --new-token-score-threshold NEW_TOKEN_SCORE_THRESHOLD
                        Only predicted tokens with a score above
                        [factor*original_score] will be taken.
  --original-token-score-threshold ORIGINAL_TOKEN_SCORE_THRESHOLD
                        Only tokens whose prediction score is above this
                        threshold will be replaced.
  --model MODEL         Name of the huggingface fill-mask model. Choose one
                        of: dict_keys(['bert-base-cased', 'bert-large-cased',
                        'roberta-large', 'roberta-base', 'distilbert-base-
                        cased'])
  --fill-mask-mode FILL_MASK_MODE
                        Mode how the generated token will be used for
                        rewriting. Choose of: ['replace', 'insert']

```


### Rewriting using a chain of transformer models

The second method uses [huggingface's "translate" task](https://huggingface.co/transformers/task_summary.html#translation) models, which mostly are machine translation transformer models by themselves.
The idea is to transform an english sentence by applying it to a chain of such MT models of which the last again produces 
an english output sentence. This way, the input sentence can be, for example, translated into German and then back into
English, producing a distinct yet semantically similar sentence. 

This whole process can be iterated multiple times, whereas each time the generated output sentence is the 
input sentence for the next iteration (if the generated output sentence has already occured before, the iteration process
is stopped earlier). 

Of course, the foreign language of the parallel corpus should not be one of the used languages in the model chain. 
In genreal, it is conceivable that parallel corpora with low-resource language pairs could benefit from this procedure if translation 
systems are available for both languages (e.g., for basque-russian there might be very few resources,
but basque-spanish and russian-english Machine Translation systems should be more widely available). 

All command line parameters for this mode can be displayed with `python main.py translate-transform --help`:

```text
usage: main.py translate-transform [-h]
                                   [--model-chain MODEL_CHAIN [MODEL_CHAIN ...]]
                                   [--iter ITER]

optional arguments:
  -h, --help            show this help message and exit
  --model-chain MODEL_CHAIN [MODEL_CHAIN ...]
                        Model chain used for transforming the sentence step by
                        step
  --iter ITER           Number of translation iterations per sentence

```

## Examples

Read in corpus from `corpus.es` and `corpus.en` and write the generated output files to `output`. The new sentences
will be produced with a translator chain using the [Helsinki-NLP](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) and [facebook](https://huggingface.co/facebook/wmt19-ru-en) translation models for Englisch -> French -> Russian -> English:

```
> python main.py -f corpus.es -e corpus.en -o output -v translate-transform --model-chain Helsinki-NLP/opus-mt-en-fr Helsinki-NLP/opus-mt-fr-ru facebook/wmt19-ru-en
```

Replace nouns, adjectives, adverbs and verbs using the fill-mask model `bert-large-cased`. Only predicted tokens with
a score greater than 0.5 are taken into consideration.

```
python main.py -f small.es -e small.en -o output2 fill-mask --model bert-large-cased --fill-mask-mode replace --new-token-score-threshold 0.5
```

The content of the directory `output2/` is then as follows:

```
❯ ls -l
insgesamt 60
-rw-r--r-- 1 daniel users   251 24. Sep 15:21 config.json
-rw-r--r-- 1 daniel users 20895 24. Sep 15:26 english.txt
-rw-r--r-- 1 daniel users 22270 24. Sep 15:26 foreign.txt
```

The `config.json` file contains the specified command line parameters (for later reference), `foreign.json` contains 
unmodified copies of the original foreign (spanish) sentences and `english.txt` contains the corresponding generated
english sentences.
