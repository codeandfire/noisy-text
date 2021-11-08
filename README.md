## Setup

To setup your environment, run the Bash script `setup.sh`.
This downloads/installs all the necessary Python packages, pretrained models, and datasets.

You need to create a virtual environment named `env` prior to running this script:
```
$ python3 -m venv env
$ source env/bin/activate
```

Then run it as:
```
$ ./setup.sh
```

More details on this script can be obtained using:
```
$ ./setup.sh -h
```

## Challenge Test Sets

First, we split the code-mixed dataset into mostly Hindi, mixed Hindi-English, and mostly English subsets:
```python
>>> from split_dataset import SplitLang
>>> sp = SplitLang()
>>> sp.filter(0, 0.2, preview=True, testset_name='codemix-hin')
>>> sp.filter(0.4, 0.6, preview=True, testset_name='codemix-mixed')
>>> sp.filter(0.8, 1, preview=True, testset_name='codemix-eng')
```
The last two commands respectively create the CM and CE test sets described in the paper.

Then, we train the bilingual character LSTM, and obtain perplexity values for the tweet samples:
```
$ python3 char_lstm.py --samples '100000' --hidden-size '100' --bidirectional
```

Next:
```python
>>> from split_dataset import SplitPerplexity
```

The following creates the ME-E and ME-C datasets respectively:
```python
>>> sp = SplitPerplexity('mono-eng')
>>> sp.filter(1.0000, 1.0012, preview=True, testset_name='mono-eng-easy')
>>> sp.filter(1.01, 1.1, preview=True, testset_name='mono-eng-challenge')
```

Finally, the CH-E and CH-C datasets are created using:
```python
>>> sp = SplitPerplexity('codemix-hin')
>>> sp.filter(1.000, 1.020, preview=True, testset_name='codemix-hin-easy')
>>> sp.filter(1.2, 1.5, preview=True, testset_name='codemix-hin-challenge')
```

## Models

#### Baseline system

The baseline system is a Bag-of-Words + TFIDF model.

The results in the ME scenario can be reproduced using:
```
$ python3 bow_tfidf.py 'mono-eng' --l1-penalty '4.0' --bow-cutoff '1'
$ python3 bow_tfidf.py 'mono-eng' --final-run
```

And in the C scenario:
```
$ python3 bow_tfidf.py 'codemix' --l1-penalty '2.0' --bow-cutoff '3'
$ python3 bow_tfidf.py 'codemix' --final-run
```

#### BERT and Related Models

To ensure that the code is working, run with the `--debug` flag. For example:
```
$ python3 bert_classifier.py 'mono-eng' --debug
```
This checks the fine-tuning scenario. Check the no-finetuning scenario with
```
$ python3 bert_classifier.py 'mono-eng' --no-finetune --debug
```
You can also check another model and another dataset. For example:
```
$ python3 bert_classifier.py 'codemix' --model 'indicbert' --debug
```

To reproduce the results in the paper:

BERT with fine-tuning in the ME scenario:
```
$ python3 bert_classifier.py 'mono-eng' --model 'bert' --learning-rate '5e-5' --batch-size '128'
$ python3 bert_classifier.py 'mono-eng' --model 'bert' --final-run
```

BERT without fine-tuning in the ME scenario:
```
$ python3 bert_classifier.py 'mono-eng' --model 'bert' --no-finetune --learning-rate '0.05' --batch-size '256'
$ python3 bert_classifier.py 'mono-eng' --model 'bert' --no-finetune --final-run
```

BERTweet with fine-tuning in the ME scenario:
```
$ python3 bert_classifier.py 'mono-eng' --model 'bertweet' --learning-rate '5e-5' --batch-size '128'
$ python3 bert_classifier.py 'mono-eng' --model 'bertweet' --final-run
```

BERTweet without fine-tuning in the ME scenario:
```
$ python3 bert_classifier.py 'mono-eng' --model 'bertweet' --no-finetune --learning-rate '0.05' --batch-size '256'
$ python3 bert_classifier.py 'mono-eng' --model 'bertweet' --no-finetune --final-run
```

BERT with fine-tuning in the C scenario:
```
$ python3 bert_classifier.py 'codemix' --model 'bert' --learning-rate '5e-5' --batch-size '32'
$ python3 bert_classifier.py 'codemix' --model 'bert' --final-run
```

BERT without fine-tuning in the C scenario:
```
$ python3 bert_classifier.py 'codemix' --model 'bert' --no-finetune --learning-rate '0.1' --batch-size '256' --epochs '2'
$ python3 bert_classifier.py 'codemix' --model 'bert' --no-finetune --final-run
```

HindiBERT with fine-tuning in the C scenario:
```
$ python3 bert_classifier.py 'codemix' --model 'hindibert' --learning-rate '5e-5' --batch-size '128' --epochs '2'
$ python3 bert_classifier.py 'codemix' --model 'hindibert' --final-run
```

HindiBERT without fine-tuning in the C scenario:
```
$ python3 bert_classifier.py 'codemix' --model 'hindibert' --no-finetune --learning-rate '0.01' --batch-size '256' --epochs '2'
$ python3 bert_classifier.py 'codemix' --model 'hindibert' --no-finetune --final-run
```

IndicBERT with fine-tuning in the C scenario:
```
$ python3 bert_classifier.py 'codemix' --model 'indicbert' --learning-rate '1e-5' --batch-size '32' --epochs '2'
$ python3 bert_classifier.py 'codemix' --model 'indicbert' --final-run
```

IndicBERT without fine-tuning in the C scenario:
```
$ python3 bert_classifier.py 'codemix' --model 'indicbert' --no-finetune --learning-rate '0.01' --batch-size '256' --epochs '2'
$ python3 bert_classifier.py 'codemix' --model 'indicbert' --no-finetune --final-run
```

#### Word Embedding Models

You can again use the `--debug` flag to check that everything is working.

fastText without fine-tuning in the ME scenario:
```
$ python3 fasttext_classifier.py 'mono-eng' --model 'fasttext' --learning-rate '0.05' --batch-size '256' --epochs '2'
$ python3 fasttext_classifier.py 'mono-eng' --model 'fasttext' --final-run
```

fastText without fine-tuning in the C scenario:
```
$ python3 fasttext_classifier.py 'codemix' --model 'fasttext' --learning-rate '0.05' --batch-size '256'
$ python3 fasttext_classifier.py 'codemix' --model 'fasttext' --final-run
```

IndicFT without fine-tuning in the C scenario:
```
$ python3 fasttext_classifier.py 'codemix' --model 'indicft' --learning-rate '0.05' --batch-size '256'
$ python3 fasttext_classifier.py 'codemix' --model 'indicft' --final-run
```
