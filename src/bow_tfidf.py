"""Softmax classifier using bag-of-words features with TF-IDF reweighting.

Baseline system.
"""

import argparse

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import utils


def _analyzer(s):
    return s.split(' ')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', type=str, choices=['mono-eng', 'codemix'], required=True,
        help='dataset to run model on'
    )
    parser.add_argument(
        '--final-run', action='store_true', default=False,
        help='do not train the model, run the trained model on test sets'
    )
    parser.add_argument(
        '--bow-cutoff', type=int, default=1, help='count cutoff for bag-of-words'
    )
    parser.add_argument(
        '--l1-penalty', type=float, default=1.0, help='l1 regularization penalty'
    )
    args = parser.parse_args()

    if not args.final_run:
        
        # load train and dev datasets
        if args.dataset == 'mono-eng':
            train = utils.load_eng_tweets_dataset(split='train')
            dev = utils.load_eng_tweets_dataset(split='dev')

        elif args.dataset == 'codemix':
            train = utils.load_hin_eng_tweets_dataset(split='train')
            dev = utils.load_hin_eng_tweets_dataset(split='dev')

        preprocess_datasets = [train, dev]

    else:

        # load test sets
        if args.dataset == 'mono-eng':
            overall = utils.load_eng_tweets_dataset(split='test')
            test_set_names = ['mono-eng-easy', 'mono-eng-challenge']

        elif args.dataset == 'codemix':
            overall = utils.load_hin_eng_tweets_dataset(split='test')
            test_set_names = [
                'codemix-hin-easy',
                'codemix-hin-challenge',
                'codemix-mixed',
                'codemix-eng'
            ]

        test_sets = []
        for name in test_set_names:
            test_sets.append(utils.load_subset(name, overall[:]))

        test_sets.insert(0, overall[:])
        test_set_names.insert(0, 'overall')

        preprocess_datasets = test_sets


    # markers to denote where each dataset starts and ends.
    markers = []
    for dataset in preprocess_datasets:
        try:
            markers.append(markers[-1] + len(dataset))
        except IndexError:
            markers.append(len(dataset))

    # preprocess all tweets in all datasets.
    tweets = utils.preprocess_tweets(
        [t['text'] for dataset in preprocess_datasets for t in dataset],
        tokenize=True,
        mask_user=True,
        mask_httpurl=True,
        mask_hashtag=False,
        mask_emoji=False,
        emoji_to_text=False
    )

    # assign the preprocessed tweets to their corresponding entries in the
    # datasets.
    for m in range(len(markers)):
        if m == 0:
            start, end = 0, markers[m]
        else:
            start, end = markers[m-1], markers[m]

        for i in range(start, end):

            # join the tokenized tweets by whitespace, for the sake of sklearn's
            # CountVectorizer which expects a list of strings.
            # _analyzer splits on this whitespace, so we get the original tokens
            # back.
            preprocess_datasets[m][i-start]['text'] = ' '.join(tweets[i])


    if not args.final_run:

        # model definition
        model = Pipeline([
            ('bow', CountVectorizer(min_df=args.bow_cutoff, analyzer=_analyzer)),
            ('tfidf', TfidfTransformer()),
            (
                'classifier',
                LogisticRegression(
                    fit_intercept=True,
                    penalty='l1',
                    C=args.l1_penalty,
                    solver='liblinear'
                )
            )
        ])

        # train the model
        X_train, y_train = [t['text'] for t in train], [t['label'] for t in train]
        model.fit(X_train, y_train)

        # save the model
        filename = 'bow-tfidf-' + args.dataset + '.pkl'
        joblib.dump(model, filename)

        test_sets = {'dev': dev}

    else:

        # load the saved model
        filename = 'bow-tfidf-' + args.dataset + '.pkl'
        model = joblib.load(filename)

        test_sets = {name: test for name, test in zip(test_set_names, test_sets)}

    # test the model
    for name, test in test_sets.items():

        X_test, y_test = [t['text'] for t in test], [t['label'] for t in test]
        pred = model.predict(X_test)

        print("Test set '{}'".format(name))
        print(classification_report(y_test, pred, digits=3))
