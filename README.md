# Parallel Corpus Augmentation for Machine Translation

This repository contains a python script that uses huggingface for augmenting 
the english side of a parallel corpus.

# Installation

This needs python >= 3.8. Requirements must be installed with `pip install -r requirements.txt`. 

## Usage
Usage instructions can be printed with:

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

## Examples

Read in corpus from `corpus.es` and `corpus.en` and write the generated output files to `output`. The new sentences
will be produced with a translator chain using the [https://huggingface.co/Helsinki-NLP/opus-mt-en-fr](Helsinki-NLP) and [facebook](https://huggingface.co/facebook/wmt19-ru-en) translation models for Englisch -> French -> Russian -> English:

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