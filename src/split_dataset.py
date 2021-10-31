"""Helper classes to split datasets on basis of (a) language (b) perplexity value."""

import random

import utils


class SplitDatasetLang:
    """Helper class to split the code-mixed Hindi-English dataset on basis of
    fraction of words labelled as English/Hindi."""

    def __init__(self):

        self._dataset = utils.load_hin_eng_tweets_dataset(
            split='train', lang_labels=True
        )
        self._dataset.extend(utils.load_hin_eng_tweets_dataset(
            split='dev', lang_labels=True
        ))
        self._dataset.extend(utils.load_hin_eng_tweets_dataset(
            split='test', lang_labels=True
        ))

        tweets = [d['text'] for d in self._dataset]
        self._orig_tweets = tweets[:]

        # preprocess the tweets
        # this is necessary because we don't want to count language labels
        # assigned to user mentions, URLs and emojis.
        tweets = utils.preprocess_tweets(
            tweets,
            tokenize=True,
            mask_user=True,
            mask_httpurl=True,
            mask_hashtag=False,
            mask_emoji=True
        )

        for t in range(len(tweets)):

            # strip user mentions, URLs, emojis, the leading # of hashtags
            tweets[t] = [
                w.replace('#', '') for w in tweets[t]
                if w not in [
                    utils.USER_SYMBOL, utils.HTTPURL_SYMBOL, utils.EMOJI_SYMBOL
                ]
            ]

            # adjust the language labels to match the tokens
            self._dataset[t]['lang_labels'] = [
                self._dataset[t]['lang_labels'].get(w) for w in tweets[t]
            ]

    @staticmethod
    def _frac_lang_labels(lang_labels):
        """Fraction of English/Hindi language labels in a tweet."""

        eng_count, hin_count = 0, 0
        for l in lang_labels:
            if l == utils.LANG_ENG:
                eng_count += 1
            elif l == utils.LANG_HIN:
                hin_count += 1

        try:
            return (eng_count/len(lang_labels), hin_count/len(lang_labels))

        # there is at least one bad tweet with no text
        except ZeroDivisionError:
            return 0, 0

    def filter(
        self,
        eng_lower,
        eng_upper,
        hin_lower=None,
        hin_upper=None,
        preview=True,
        save_filename=None
    ):
        """Filter the code-mixed dataset.

        Arguments
        ---------
            eng_lower : float
                lower bound on proportion of English labels

            eng_upper : float
                upper bound on proportion of English labels

            hin_lower : float
                lower bound on proportion of Hindi labels.
                if None, it is calculated so as to complement the proportion
                of English labels.

            hin_upper : float
                upper bound on proportion of Hindi labels.
                if None, it is calculated so as to complement the proportion
                of English labels.

            preview : bool
                preview the filtered subset.
                returns a random sample of 20 tweets from the filtered subset.

            save_filename : str
                name of file to which the filtered subset is written.
                if None, the filtered subset is not saved.
        """

        # calculate the bounds on proportions of Hindi labels, if not provided
        hin_lower = (1-eng_upper) if hin_lower is None else hin_lower
        hin_upper = (1-eng_lower) if hin_upper is None else hin_upper

        # prepare the subset
        # indices of elements in the dataset that fall into the subset are
        # recorded in `idxs` for previewing of the tweets.
        subset, idxs = [], []

        for i, d in enumerate(self._dataset):

            eng_frac, hin_frac = self._frac_lang_labels(d['lang_labels'])

            if (eng_lower <= eng_frac <= eng_upper) and (
                    hin_lower <= hin_frac <= hin_upper
            ):
                subset.append(d)
                idxs.append(i)

        if save_filename is not None:
            utils.write_subset(subset, save_filename)
        
        if preview:
            sample = [self._orig_tweets[random.choice(idxs)] for _ in range(20)]
            return sample
