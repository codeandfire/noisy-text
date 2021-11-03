"""Classifier based on BERT / related models, both in fine-tuning and no fine-
tuning configurations."""

import argparse
import copy
import logging

from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import settings
import utils


class BERTDataset(Dataset):
    """Dataset definition required by PyTorch."""

    def __init__(self, dict_dataset, bert_tokenizer):
        """Initialize the dataset.

        Pass a dataset in the form of a list of dictionaries, each dictionary
        having keys 'text' and 'label' giving the tokenized text of the tweet
        and the sentiment label as an integer index respectively.
        Also pass the BERT tokenizer that needs to be applied on the tweets.
        """

        super().__init__()

        self.encoding = bert_tokenizer(
            [d['text'] for d in dict_dataset],

            # start- and end-of-sentence tokens
            add_special_tokens=True,

            # pad upto the longest sequence in the dataset
            padding='longest',

            # text is already tokenized
            is_split_into_words=True,

            return_attention_mask=True,
            return_tensors='pt'
        )
        self.labels = [d['label'] for d in dict_dataset]

    def __len__(self):

        # NOTE: len(self.encoding) does not give the right value for some
        # reason - it gives the value 3!
        # so we must use len(self.labels) and not len(self.encoding)
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.encoding.input_ids[idx],
            self.encoding.attention_mask[idx],
            self.labels[idx]
        )


class BERTClassifier(nn.Module):
    """Model definition."""

    NUM_CLASSES = 3

    def __init__(self, bert_model, finetune=True):
        super().__init__()
        self.bert_model = bert_model
        self.finetune = finetune

        if not self.finetune:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(
            in_features=self.bert_model.config.hidden_size,
            out_features=self.NUM_CLASSES
        )

    def forward(self, input_ids, attention_mask):
        bert_reps = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if self.finetune:

            # use the [CLS] token representation
            try:
                return self.classifier(bert_reps.pooler_output)

            except AttributeError:   # happens in the case of ELECTRA models
                return self.classifier(bert_reps.last_hidden_state[:, 0, :])

        # mean pool over hidden states
        return self.classifier(
            torch.mean(bert_reps.last_hidden_state, dim=1)
        )


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
        '--model', type=str, choices=[
            'bert', 'bertweet', 'hindibert', 'indicbert'
        ],
        default='bert',
        help='choice of BERT model'
    )
    parser.add_argument(
        '--no-finetune', action='store_true', default=False,
        help='do not finetune the BERT model'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=1e-5, help='learning rate'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256, help='batch size'
    )
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument(
        '--record-loss', type=int, default=5,
        help='record training loss after every given number of batches'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='run in debug mode, i.e. take only 50 samples for train/dev. (not applicable if --final-run)'
    )

    args = parser.parse_args()


    if args.model == 'bert':
        model_name, cache_dir = 'bert-base-cased', settings.BERT_ROOT
    elif args.model == 'bertweet':
        model_name, cache_dir = 'vinai/bertweet-base', settings.BERTWEET_ROOT
    elif args.model == 'hindibert':
        model_name, cache_dir = 'monsoon-nlp/hindi-bert', settings.HINDIBERT_ROOT
    elif args.model == 'indicbert':
        model_name, cache_dir = 'ai4bharat/indic-bert', settings.INDICBERT_ROOT

    # load the tokenizer
    if args.model == 'bertweet':

        # don't perform normalization.
        # this is crucial because BERTweet's normalization is already performed
        # as part of the preprocessing we perform later.
        bert_tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, normalization=False
        )

    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )

    # load the BERT model
    bert_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)


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

        if args.model == 'hindibert':
            # back-transliterate each tweet completely.
            tweets = [utils.back_transliterate(t) for t in tweets]

        elif args.model == 'indicbert':

            # get language labels for all tweets in all datasets.
            lang_labels = [
                t['lang_labels'] for dataset in preprocess_datasets for t in dataset
            ]

            # adjust language labels to match tokens of tweets.
            for i in range(len(tweets)):
                lang_labels[i] = [lang_labels[i].get(t) for t in tweets[i]]

            # back-transliterate only those tokens labelled as Hindi.
            tweets = [
                utils.back_transliterate(t, lang_labels=l)
                for t, l in zip(tweets, lang_labels)
            ]


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
    filename = '{}-{}-'.format(args.model, args.dataset)
    filename += 'no-finetune' if args.no_finetune else 'finetune'
    filename += '.pt'


    model = BERTClassifier(bert_model, finetune=(not args.no_finetune))

    if args.final_run:

        # load the saved model
        model.load_state_dict(torch.load(filename))

        test_sets = {name: test for name, test in zip(test_set_names, test_sets)}

    else:

        # train the model.
        # prepare the train set.
        train = BERTDataset(train, bert_tokenizer)
        dataloader = DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True
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
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()

                scores = model(input_ids, attention_mask)

                loss = loss_fn(scores, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % args.record_loss == args.record_loss-1:
                    logging.info(
                        f'epoch {epoch_idx+1} batch {batch_idx+1}: loss = {running_loss:.6f}'
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

        test = BERTDataset(test, bert_tokenizer)
        dataloader = DataLoader(
            test,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True
        )

        true, pred = [], []

        with torch.no_grad():

            for batch in tqdm(
                dataloader,
                desc='inference',
                unit='batch'
            ):

                batch = [b.to(device, non_blocking=True) for b in batch]
                input_ids, attention_mask, labels = batch

                scores = model(input_ids, attention_mask)

                pred.extend(torch.argmax(scores, dim=1).tolist())
                true.extend(labels.tolist())

        # sentiment labels are indexes as converted by labels_to_idx
        # convert them back to labels to see them in the report
        idx_to_labels = {i: l for l, i in labels_to_idx.items()}
        true = [idx_to_labels[i] for i in true]
        pred = [idx_to_labels[i] for i in pred]

        print("Test set '{}'".format(name))
        print(classification_report(true, pred, digits=3))
