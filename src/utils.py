import csv
import os
import re
import string

from emoji import is_emoji, demojize, replace_emoji
from indictrans import Transliterator
from nltk.tokenize.casual import TweetTokenizer

import settings


LANG_ENG = 'eng'
LANG_HIN = 'hin'

START_SYMBOL = '<S>'
END_SYMBOL = '</S>'
USER_SYMBOL = '@USER'
HTTPURL_SYMBOL = 'HTTPURL'
HASHTAG_SYMBOL = '#HASHTAG'
EMOJI_SYMBOL = 'EMOJI'

_trn = Transliterator(source=LANG_ENG, target=LANG_HIN, build_lookup=True)


def load_en_tweets_dataset(split='train'):

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
        os.path.join(settings.EN_TWEETS_ROOT, filename),
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


def load_hi_en_tweets_dataset(split='train', lang_labels=False):
    
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
        os.path.join(settings.HI_EN_TWEETS_ROOT, filename),
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
            os.path.join(settings.HI_EN_TWEETS_ROOT, 'test-labels.txt'), 'r'
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

    def detokenize(text):

        # join all tokens, separate by whitespace
        text = ' '.join(text)

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

            # after detokenization, the language labels do not correspond to
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
        tweets = [replace_emoji(t, EMOJI_SYMBOL) for t in tweets]
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


def back_transliterate(text, mask=None):
    if mask is None:
        return _trn.transform(text)
    return [(_trn.transform(t) if m else t) for m, t in zip(mask, tokens)]