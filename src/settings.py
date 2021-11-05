"""Global settings for the project.

The ROOTs with value None are automatically set to the appropriate paths by
setup.sh.
"""

# BERT and co.
BERT_ROOT = None
BERTWEET_ROOT = None
HINDIBERT_ROOT = None
INDICBERT_ROOT = None

# word embeddings
FASTTEXT_ROOT = None
INDICFT_ROOT = None

# datasets
ENG_TWEETS_ROOT = None
HIN_ENG_TWEETS_ROOT = None

# canonical corpora
ENG_CORPUS_ROOT = None
HIN_CORPUS_ROOT = None

# challenge test sets
CHALLENGE_ROOT = None

TESTSET_FILENAMES = {
    'mono-eng': None,
    'mono-eng-challenge': 'mono_eng_challenge.txt',
    'mono-eng-easy': 'mono_eng_easy.txt',
    'codemix': None,
    'codemix-eng': 'codemix_eng.txt',
    'codemix-mixed':  'codemix_mixed.txt',
    'codemix-hin': 'codemix_hin.txt',
    'codemix-hin-challenge': 'codemix_hin_challenge.txt',
    'codemix-hin-easy': 'codemix_hin_easy.txt'
}

PERPLEXITIES_FILE = 'perplexities.txt'
