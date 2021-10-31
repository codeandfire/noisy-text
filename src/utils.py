"""Utility functions."""

import bisect
import csv
import os
import re
import string

from emoji import is_emoji, demojize, replace_emoji
from indictrans import Transliterator
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

import settings


# language labels

LANG_ENG = 'eng'
LANG_HIN = 'hin'


# special tokens

START_SYMBOL = '<S>'
END_SYMBOL = '</S>'
USER_SYMBOL = '@USER'
HTTPURL_SYMBOL = 'HTTPURL'
HASHTAG_SYMBOL = '#HASHTAG'
EMOJI_SYMBOL = 'EMOJI'


# for back-transliteration from Roman script to Devanagari
_trn = Transliterator(source=LANG_ENG, target=LANG_HIN, build_lookup=True)


def load_eng_tweets_dataset(split='train'):
    """Load English tweets dataset (SemEval-2017 Task 4 Subtask A dataset).

    Specify which split to load. Default is 'train', you can also specify
    'dev' or 'test'.

    Returns the dataset as a list of dictionaries, with keys 'tweet_id',
    'text' and 'label'.
    """

    if split == 'train' or split == 'dev':
        filename = 'train-dev.txt'
    elif split == 'test':
        filename = 'test.txt'
    else:
        raise ValueError(
            'Unknown split {}: split must be one of {}'.format(
                split, "'train', 'dev' or 'test'"
            )
        )


    # reading the dataset.
    # it is a tab-separated file with 3 fields corresponding to the tweet ID,
    # sentiment label, and text of the tweet.
    # utf-8 encoding is used due to the presence of emojis and other Unicode
    # characters.
    # (the newline='' argument is required by csv.DictReader)

    with open(
        os.path.join(settings.ENG_TWEETS_ROOT, filename),
        'r',
        encoding='utf-8',
        newline=''
    ) as dataset_file:

        reader = csv.DictReader(
            dataset_file,
            delimiter='\t',
            fieldnames=['tweet_id', 'label', 'text'],
            quoting=csv.QUOTE_NONE
        )

        dataset = list(reader)

    # train-dev split is not predefined in the dataset, use an 85-15 split.
    if split == 'train':
        dataset = dataset[: int(0.85 * len(dataset))]
    elif split == 'dev':
        dataset = dataset[int(0.85 * len(dataset)) :]

    return dataset


def load_hin_eng_tweets_dataset(split='train', lang_labels=False):
    """Load Hindi-English tweets dataset (SemEval-2020 Task 9 Dataset).

    Specify which split to load. Default is 'train', you can also specify
    'dev' or 'test'.

    If word-level language labels are required, specify lang_labels=True.

    Returns the dataset as a list of dictionaries, with keys 'tweet_id',
    'text' and 'label', and also 'lang_labels' if lang_labels=True.
    """
    
    if split == 'train':
        filename = 'train.txt'
    elif split == 'dev':
        filename = 'dev.txt'
    elif split == 'test':
        filename = 'test.txt'
    else:
        raise ValueError(
            'Unknown split {}: split must be one of {}'.format(
                split, "'train', 'dev' or 'test'"
            )
        )


    # reading the dataset.
    # the dataset is in CoNLL format with a tweet ID and sentiment label.
    # the tweets are already tokenized; token-level language labels are also
    # provided.
    # utf-8 encoding is used due to the presence of emojis and other Unicode
    # characters.

    with open(
        os.path.join(settings.HIN_ENG_TWEETS_ROOT, filename),
        'r',
        encoding='utf-8'
    ) as dataset_file:

        dataset = []

        record = None

        for line in dataset_file.readlines():

            # entries are separated by tabs
            fields = line.strip().split('\t')

            # this is a blank line, indicates the end of the previous record
            if len(fields) == 1 and fields[0] == '':
                dataset.append(record)
                record = None

            # this is the start of a new record
            elif fields[0] == 'meta' and record is None:
                record = {
                    'tweet_id': None,
                    'label': None,
                    'text': [],
                    'lang_labels': []
                }

                record['tweet_id'] = fields[1]

                # in the test split, sentiment labels are provided in a separate
                # file.
                if split != 'test':
                    record['label'] = fields[2]

            else:

                # append tokens and language labels to the current record
                try:
                    record['text'].append(fields[0])
                    record['lang_labels'].append(fields[1])

                # this can happen in the case of some stray bad lines
                except IndexError:
                    continue

    if split == 'test':

        # load the sentiment labels of the test split
        with open(
            os.path.join(settings.HIN_ENG_TWEETS_ROOT, 'test-labels.txt'), 'r'
        ) as f:

            # the first line is a header, ignore it
            f.readline()

            # each line is a tweet ID and the corresponding sentiment label,
            # separated by a comma.
            for line in f.readlines():
                entries = line.strip().split(',')
                tweet_id, label = entries[0], entries[1]
                
                for d in range(len(dataset)):
                    if dataset[d]['tweet_id'] == tweet_id:
                        dataset[d]['label'] = label
                        break


    # as mentioned previously, the tweets are already tokenized.
    # unfortunately, the tokenization is not very convenient for our purposes.
    # the following code detokenizes the tweets back to a string.

    # some regular expressions applied on the detokenized text
    user_regexp = re.compile(r'@\s(\S+)')
    underscore_regexp = re.compile(r'(\S+)\s_\s(\S+)')
    hashtag_regexp = re.compile(r'#\s(\S+)')
    httpurl_regexp = re.compile(
        r'https\s//\s(t\s\.\sco|t\sco|tco)\s/\s(\w+)'
    )

    # the following detokenizer corrects spacing around punctuation marks
    detok = TreebankWordDetokenizer()

    def detokenize(text):

        text = detok.detokenize(text)

        # join all usernames, i.e. turn @ username to @username
        text = user_regexp.sub(r'@\1', text)

        # the same with underscores, i.e. turn a _ b to a_b
        text = underscore_regexp.sub(r'\1_\2', text)

        # the same with hashtags, i.e. # hashtag to #hashtag
        text = hashtag_regexp.sub(r'#\1', text)

        # in the original dataset, HTTP URLs are split up into several parts;
        # join all of them together.
        text = httpurl_regexp.sub(r'https://t.co/\2', text)

        return text


    for d in range(len(dataset)):

        if not lang_labels:
            dataset[d].pop('lang_labels')
        else:

            # language labels are provided as 'Eng' and 'Hin' in the dataset;
            # lowercase them for consistency with LANG_ENG and LANG_HIN above.
            dataset[d]['lang_labels'] = [
                l.lower() for l in dataset[d]['lang_labels']
            ]

            # after detokenization, the language labels do not correspond to the
            # same tokens since many of the tokens have been joined together.
            # a way to preserve the label information is to convert the list of
            # labels into a mapping from the original tokens to the labels.
            dataset[d]['lang_labels'] = {
                t: l for t, l in zip(dataset[d]['text'], dataset[d]['lang_labels'])
            }

        # detokenize the text
        dataset[d]['text'] = detokenize(dataset[d]['text'])

    return dataset


def preprocess_tweets(
    tweets,
    tokenize=True,
    mask_user=False,
    mask_httpurl=False,
    mask_hashtag=False,
    mask_emoji=False,
    emoji_to_text=False,
    normalize_misc_unicode=True
):
    """Preprocess tweet samples.

    Arguments
    ---------

        tweets : list of str
            the text of the tweets.

        tokenize : bool
            use NLTK's TweetTokenizer to tokenize the tweets.

        mask_user : bool
            convert user mentions to special token.

        mask_httpurl : bool
            convert HTTP URLs to special token.

        mask_hashtag  : bool
            convert hashtags to special token.

        mask_emoji : bool
            convert emojis to special token.

        emoji_to_text : bool
            utilize `emoji' package to convert emojis to textual descriptions.
            not applicable if mask_emoji=True.

        normalize_misc_unicode : bool
            normalize miscellaneous Unicode characters that can be replaced by
            regular ASCII characters.

        Returns
        -------
            a list of strings if tokenize=False, otherwise a list of lists of
            strings if tokenize=True.
    """

    # regexes to identify user mentions, HTTP URLs and hashtags
    user_regexp = re.compile(r'@\S+')
    httpurl_regexp = re.compile(r'http\S+|www\S+')
    hashtag_regexp = re.compile(r'#\S+')

    if mask_user:
        tweets = [user_regexp.sub(USER_SYMBOL, t) for t in tweets]
    if mask_httpurl:
        tweets = [httpurl_regexp.sub(HTTPURL_SYMBOL, t) for t in tweets]
    if mask_hashtag:
        tweets = [hashtag_regexp.sub(HASHTAG_SYMBOL, t) for t in tweets]

    if mask_emoji:
        tweets = [replace_emoji(t, ' '+EMOJI_SYMBOL+' ') for t in tweets]
    elif emoji_to_text:
        tweets = [demojize(t, delimiters=(' ', ' ')) for t in tweets]

    if normalize_misc_unicode:
        tweets = [
            t.replace(u"’", "'").replace(u"‘", "'").replace(u'…', '...')
            for t in tweets
        ]

    if tokenize:
        tok = TweetTokenizer()
        tweets = [tok.tokenize(t) for t in tweets]

    return tweets


def back_transliterate(text, lang_labels=None):
    """Back-transliterate from Roman script to Devanagari.

    Optionally accepts a list of language labels, in which case it back-
    transliterates only those tokens marked as LANG_HIN. In this case, `text'
    must be a list of strings (tokens) rather than a single string.
    """

    if lang_labels is None:
        return _trn.transform(text)

    translit_text = []
    for i, t in enumerate(text):
        if lang_labels[i] == LANG_HIN:
            translit_text.append(_trn.transform(t))
        else:
            translit_text.append(t)

    return translit_text


def write_subset(subset, testset_name):
    """Write a subset (of some dataset) as a test set.

    `testset_name` must be a string value, a key in the dictionary
    `settings.TESTSET_FILENAMES`.
    `subset` must be a regular dataset: a list of entries represented by
    dictionaries, each dictionary having key 'tweet_id'.
    """

    with open(
        os.path.join(
            settings.CHALLENGE_ROOT, settings.TESTSET_FILENAMES[testset_name]
        ), 'w'
    ) as f:
        f.write('\n'.join([s['tweet_id'] for s in subset]))


def load_subset(testset_name, dataset):
    """Load a subset of `dataset` containing only those of its entries that
    are present in a given test set.

    `testset_name` must be a string value, a key in the dictionary
    `settings.TESTSET_FILENAMES`.
    `dataset` must be a list of entries represented by dictionaries, each
    dictionary having key 'tweet_id'.
    """

    with open(
        os.path.join(
            settings.CHALLENGE_ROOT, settings.TESTSET_FILENAMES[testset_name]
        ), 'r'
    ) as f:
        tweet_ids = [line.strip() for line in f.readlines()]

    # using binary search; ordinary search may be slow.
    # sorting prior to using binary search.
    tweet_ids.sort()

    subset = []

    for d in dataset:
        pos = bisect.bisect_left(tweet_ids, d['tweet_id'])
        if tweet_ids[pos] == d['tweet_id']:   # found
            subset.append(d)

    return subset


def write_perplexities(dataset):
    """Write perplexities of entries in `dataset` to file.

    `dataset` must be a list of entries represented by dictionaries, each
    dictionary having the keys 'tweet_id' and 'perplexity'.
    """

    with open(
        os.path.join(settings.CHALLENGE_ROOT, settings.PERPLEXITIES_FILE),
        'w', newline=''
    ) as f:

        writer = csv.writer(f, delimiter=',')
        for d in dataset:
            writer.writerow([d['tweet_id'], '{:.8f}'.format(d['perplexity'])])


def load_perplexities(dataset):
    """Load perplexity values of entries in `dataset` from file.

    `dataset` must be a list of entries represented by dictionaries, each
    dictionary having the key 'tweet_id'.

    Returns `dataset` with the key 'perplexity' added to each entry, containing
    the perplexity value.
    """

    with open(
        os.path.join(settings.CHALLENGE_ROOT, settings.PERPLEXITIES_FILE),
        'r', newline=''
    ) as f:

        reader = csv.DictReader(
            f,
            fieldnames=['tweet_id', 'perplexity'],
            delimiter=','
        )
        perps = list(reader)

    # sorting to search efficiently.
    perps.sort(key=lambda p: p['tweet_id'])
    dataset.sort(key=lambda d: d['tweet_id'])

    c = 0
    for p in perps:
        try:
            if p['tweet_id'] == dataset[c]['tweet_id']:
                dataset[c]['perplexity'] = p['perplexity']
                c = c + 1
        except IndexError:   # c is out of bounds
            break

    return dataset
