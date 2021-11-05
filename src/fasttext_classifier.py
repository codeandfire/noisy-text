"""Classifier using fastText embeddings as (a) vector-averaged inputs and (b)
inputs to LSTM at each timestep.
"""

import argparse
import copy
import logging
import os

import fasttext
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import settings
import utils


class FasttextDataset(Dataset):

    def __init__(self, texts, labels, fasttext_model):
        super().__init__()

        self.vecavgs = [
            np.mean([
                fasttext_model.get_word_vector(w) for w in t
            ], axis=0)
            for t in texts
        ]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def  __getitem__(self, idx):
        return self.vecavgs[idx], self.labels[idx]


class LinearClassifier(nn.Module):

    NUM_CLASSES = 3

    def __init__(self, input_size=300):
        super().__init__()

        self.classifier = nn.Linear(
            in_features=input_size, out_features=self.NUM_CLASSES
        )

    def forward(self, vecavgs):
        return self.classifier(vecavgs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'dataset', type=str, choices=['mono-eng', 'codemix'],
        help='dataset to run model on'
    )
    parser.add_argument(
        '--final-run', action='store_true', default=False,
        help='do not train the model, run the trained model on test sets'
    )
    parser.add_argument(
        '--model', type=str, choices=['fasttext', 'indicft'], default='fasttext',
        help='choice of fasttext model'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256, help='batch size'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-5, help='learning rate'
    )
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument(
        '--record-loss', type=int, default=100,
        help='record training loss after every given number of updates'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='run in debug mode, i.e. take only 50 samples for train/dev. (not applicable if --final-run)'
    )

    args = parser.parse_args()

    if args.model == 'fasttext':
        fasttext_model = fasttext.load_model(
            os.path.join(settings.FASTTEXT_ROOT, 'crawl-300d-2M-subword.bin')
        )
    elif args.model == 'indicft':
        fasttext_model = fasttext.load_model(
            os.path.join(settings.INDICFT_ROOT, 'indicnlp.ft.hi.300.bin')
        )


    if not args.final_run:

        # load train and dev sets
        if args.dataset == 'mono-eng':
            train = utils.load_eng_tweets_dataset(split='train')
            dev = utils.load_eng_tweets_dataset(split='dev')

        elif args.dataset == 'codemix':
            train = utils.load_hin_eng_tweets_dataset(
                split='train', lang_labels=True
            )
            dev = utils.load_hin_eng_tweets_dataset(
                split='dev', lang_labels=True
            )

        if args.debug:

            # use only the first 50 samples!
            train, dev = train[:50], dev[:50]

        preprocess_datasets = [train, dev]

    else:

        # load test sets
        if args.dataset == 'mono-eng':
            overall = utils.load_eng_tweets_dataset(split='test')
            test_set_names = ['mono-eng-easy', 'mono-eng-challenge']

        elif args.dataset == 'codemix':
            overall = utils.load_hin_eng_tweets_dataset(
                split='test', lang_labels=True
            )
            test_set_names = [
                'codemix-hin-easy',
                'codemix-hin-challenge',
                'codemix-mixed',
                'codemix-eng'
            ]

        test_sets = []
        for name in test_set_names:
            test_sets.append(utils.load_subset(name, copy.deepcopy(overall)))

        # 'overall' is also a test set!
        test_sets.insert(0, overall)
        test_set_names.insert(0, 'overall')

        preprocess_datasets = test_sets


    labels_to_idx = {'positive': 0, 'neutral': 1, 'negative': 2}

    for dataset in preprocess_datasets:

        tweets = utils.preprocess_tweets(
            [d['text'] for d in dataset],
            tokenize=True,
            mask_user=True,
            mask_httpurl=True,
            mask_hashtag=False,
            mask_emoji=False,
            emoji_to_text=True
        )

        if args.model == 'indicft':
            # back-transliterate each tweet completely.
            tweets = [utils.back_transliterate(t) for t in tweets]

        for d in range(len(dataset)):
            dataset[d]['text'] = tweets[d]

            # convert sentiment labels to indices
            try:
                dataset[d]['label'] = labels_to_idx[dataset[d]['label']]
            except KeyError:

                # there is (at least) one bad label which is neither 'positive',
                # 'negative' nor 'neutral'
                # setting it to 'positive' here; it doesn't really matter which
                # class is chosen since it is (or seems to be) only one sample.
                dataset[d]['label'] = labels_to_idx['positive']


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # filename which the model will be (i) saved to, if training or (ii) loaded
    # from, if final run.
    filename = '{}-{}.pt'.format(args.model, args.dataset)


    model = LinearClassifier()

    if args.final_run:

        # load the saved model
        model.load_state_dict(torch.load(filename))

        test_sets = {name: test for name, test in zip(test_set_names, test_sets)}

    else:

        # train the model.
        # prepare the train set.
        train = FasttextDataset(
            [t['text'] for t in train],
            [t['label'] for t in train],
            fasttext_model
        )
        dataloader = DataLoader(
            train,
            shuffle=True,
            pin_memory=True,
            batch_size=args.batch_size
        )

        model.to(device)

        loss_fn = CrossEntropyLoss(reduction='mean')
        optimizer = Adam(model.parameters(), lr=args.learning_rate)

        # set up logging
        logging.basicConfig(filename='train.log', filemode='w', level=logging.INFO)
        print('Periodic record of training loss will be written to train.log.')


        # training loop

        for epoch_idx in range(args.epochs):

            if args.epochs > 1:
                print('Epoch {}'.format(epoch_idx+1))
            running_loss = 0.0

            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc='training',
                total=len(dataloader),
                unit='batch'
            ):

                batch = [b.to(device, non_blocking=True) for b in batch]
                vecavgs, labels = batch

                optimizer.zero_grad()

                scores = model(vecavgs)

                loss = loss_fn(scores, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % args.record_loss == args.record_loss-1:
                    logging.info(
                        f'epoch {epoch_idx+1} batch {batch_ids+1}: loss = {running_loss:.6f}'
                    )
                    running_loss = 0.0

        # save the model
        torch.save(model.state_dict(), filename)

        test_sets = {'dev': dev}


    model.to(device)

    # put the model in inference mode
    model.eval()

    # test the model
    for name, test in test_sets.items():

        test = FasttextDataset(
            [t['text'] for t in test],
            [t['label'] for t in test],
            fasttext_model
        )
        dataloader = DataLoader(
            test,
            shuffle=False,
            pin_memory=True,
            batch_size=args.batch_size
        )

        true, pred = [], []

        with torch.no_grad():

            for batch in tqdm(
                dataloader, desc='inference', unit='batch'
            ):
                batch = [b.to(device, non_blocking=True) for b in batch]
                vecavgs, labels = batch

                scores = model(vecavgs)

                pred.extend(torch.argmax(scores, dim=1).tolist())
                true.extend(labels.tolist())

        # sentiment labels are indexes as converted by labels_to_idx
        # convert them back to labels to see them in the report
        idx_to_labels = {i: l for l, i in labels_to_idx.items()}
        true = [idx_to_labels[i] for i in true]
        pred = [idx_to_labels[i] for i in pred]

        print("Test set '{}'".format(name))
        print(classification_report(true, pred, digits=3))
