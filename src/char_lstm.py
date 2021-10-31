"""Character-level LSTM trained on canonical Hindi and English corpora.

The perplexity of this model can be considered as a measure of how "noisy" a
given English/Hindi tweet sample is.
"""

import argparse
import logging
import os
import random

from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot, softmax
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import settings
import utils


# sequences of variable length have to be padded, so that they are of the same
# length, when batched together.
# PADDING_TOKEN is the literal value used for padding the tensors, i.e. 0.
PADDING_TOKEN = 0


# maximum allowed length of clean sample sequences (not tweets);
# sequences beyond this length will be truncated.
MAX_SEQ_LENGTH = 225


class CharDataset(Dataset):
    """Dataset definition required by PyTorch."""

    def __init__(self, corpus_as_idxs):
        """Initialize the dataset.

        Pass the tokenized corpus as a list of lists of integers, representing
        indices of the individual characters.
        """

        super().__init__()
        self.corpus_as_idxs = corpus_as_idxs

    def __len__(self):
        return len(self.corpus_as_idxs)

    def __getitem__(self, idx):

        # convert characters to indices
        seq = self.corpus_as_idxs[idx]

        # objective is to predict the next word, accordingly prepare input and
        # expected output sequences.
        input_seq, output_seq = seq[:-1], seq[1:]

        # length of the input/output sequence
        seq_length = len(seq) - 1

        return input_seq, seq_length, output_seq

    @staticmethod
    def collate_fn(batch):

        # unpack the individual elements of the batch
        input_seqs, seq_lengths, output_seqs = zip(*batch)

        # convert to PyTorch tensors
        input_seqs = [torch.tensor(seq) for seq in input_seqs]
        seq_lengths = torch.tensor(seq_lengths)
        output_seqs = [torch.tensor(seq) for seq in output_seqs]

        # pad the variable length input and output sequences
        input_seqs = pad_sequence(
            input_seqs,
            batch_first=True,
            padding_value=PADDING_TOKEN
        )
        output_seqs = pad_sequence(
            output_seqs,
            batch_first=True,
            padding_value=PADDING_TOKEN
        )

        return input_seqs, seq_lengths, output_seqs


class CharLSTM(nn.Module):
    """Model definition."""

    def __init__(
        self,
        charset_size,
        hidden_size,
        num_layers=1,
        dropout=0,
        bidirectional=False
    ):
        """Initialize the model.

        Pass the size of character set (charset), the hidden size of the LSTM,
        the number of layers in the LSTM, dropout used in the LSTM, and
        whether the LSTM should be bidirectional or not.
        """

        super().__init__()

        # one-hot vectors are used as character embeddings
        # padding_idx=PADDING_TOKEN reserves the one-hot vector at position
        # 0 for the padding token.
        self.embedding = nn.Embedding.from_pretrained(
            one_hot(torch.arange(charset_size)).to(dtype=torch.float32),
            freeze=True,
            padding_idx=PADDING_TOKEN
        )

        self.lstm = nn.LSTM(
            input_size=charset_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_layer = nn.Linear(
            in_features=(2*hidden_size) if bidirectional else hidden_size,
            out_features=charset_size
        )

    def forward(self, seqs, seq_lengths):
        """Forward pass.

        Padded sequences should be passed as input.
        Lengths of the sequences should also be passed, as a tensor.
        """

        seqs = self.embedding(seqs)

        # convert padded sequences to packed sequences for efficiency
        seqs = pack_padded_sequence(
            seqs,
            lengths=seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        seqs, _ = self.lstm(seqs)

        # convert packed sequences to padded sequences in order to pass them
        # through output feedforward layer
        seqs, _ = pad_packed_sequence(
            seqs,
            batch_first=True,
            padding_value=PADDING_TOKEN
        )

        seqs = self.output_layer(seqs)
        return seqs


def char_perplexity(pred, output_seqs, seq_lengths):
    """Function to find the perplexity values.

    Pass the tensor containing the un-normalized scores output by the model as
    pred.
    Additionally, pass the expected output sequences as well, along with the
    sequence lengths.
    """

    # convert the un-normalized scores into normalized probability values
    pred = softmax(pred, dim=2)

    perps = []

    # for each sequence in the batch
    for s in range(len(output_seqs)):

        # extract the probability values corresponding to the expected output
        # sequence.
        # the sequence lengths are used to ensure that the entries
        # corresponding to any padding tokens are ignored.
        probs = pred[
            s, torch.arange(seq_lengths[s]), output_seqs[s][:seq_lengths[s]]
        ]


        # recall that perplexity = (p1 * p2 * ... * pn) ** (-1/n)
        # so, log perplexity = -1/n * (log p1 + log p2 + ... + log pn)
        # we calculate the perplexity value using this log scale because it is
        # more accurate.
        # (if we use the original formula, the product of many small probability
        # values gives the value 0 because of lack of precision.)
        # then, the exponentiation of this log perplexity value gives back the
        # original perplexity value.

        probs = torch.log(probs)
        perps.append(
            torch.exp(-1/seq_lengths[s] * torch.sum(probs)).item()
        )

    return perps


# main script

if __name__ == '__main__':


    # command-line arguments

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--samples', type=int, default=10**6,
        help='number of clean samples from both the English/Hindi corpora used for training'
    )
    parser.add_argument(
        '--hidden-size', type=int, default=50, help='hidden dimensionality of LSTM'
    )
    parser.add_argument(
        '--num-layers', type=int, default=1, help='number of hidden layers in LSTM'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.0, help='dropout used in LSTM'
    )
    parser.add_argument(
        '--bidirectional', action='store_true', default=False,
        help='use a bidirectional LSTM'
    )
    parser.add_argument(
        '--learning-rate', type=int, default=1e-2, help='learning rate'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256, help='batch size'
    )
    parser.add_argument(
        '--epochs', type=int, default=1, help='training epochs'
    )
    parser.add_argument(
        '--record-loss', type=int, default=5,
        help='record training loss after every given number of batches'
    )

    args = parser.parse_args()


    # load the canonical corpora

    with open(os.path.join(settings.ENG_CORPUS_ROOT, 'corpus.txt'), 'r') as f:
        corpus = [f.readline() for _ in range(args.samples)]

    # the following detokenizer corrects spacing around punctuation marks;
    # which is present in this English corpus.
    detok = TreebankWordDetokenizer()
    corpus = [detok.detokenize(doc.split(' ')) for doc in corpus]

    with open(
        os.path.join(settings.HIN_CORPUS_ROOT, 'corpus.txt'), 'r',
        encoding='utf-8'     # for Devanagari characters
    ) as f:
        corpus.extend([f.readline() for _ in range(args.samples)])

    for c in range(len(corpus)):
        
        # truncate sequences to a maximum length
        corpus[c] = corpus[c][:MAX_SEQ_LENGTH]

        # tokenize into characters, add start- and end-of-sentence symbols
        corpus[c] = [utils.START_SYMBOL] + list(corpus[c]) + [utils.END_SYMBOL]

    # the set of unique characters; the "vocabulary" of a character-level model
    charset = sorted(set([c for doc in corpus for c in doc]))

    # convert characters to indices.
    # adding 1 to the indices to accommodate for the padding token at
    # position 0.
    char_to_idx = {c: i+1 for i, c in enumerate(charset)}
    corpus = [[char_to_idx[c] for c in doc] for doc in corpus]


    # load the noisy tweet samples
    
    dataset = []

    # markers to mark subsets of the dataset that have to be back-transliterated
    markers = {
        'translit-all': None,
        'translit-only-hin': None
    }

    dataset.extend(utils.load_eng_tweets_dataset(split='train'))
    dataset.extend(utils.load_eng_tweets_dataset(split='dev'))
    dataset.extend(utils.load_eng_tweets_dataset(split='test'))

    codemix = utils.load_hin_eng_tweets_dataset(split='train', lang_labels=True)
    codemix.extend(utils.load_hin_eng_tweets_dataset(
        split='dev', lang_labels=True
    ))
    codemix.extend(utils.load_hin_eng_tweets_dataset(
        split='test', lang_labels=True
    ))

    dataset.extend(utils.load_subset(settings.CODEMIX_ALL_ENG_FILE, codemix))

    markers['translit-all'] = len(dataset)
    dataset.extend(utils.load_subset(settings.CODEMIX_ALL_HIN_FILE, codemix))

    markers['translit-only-hin'] = len(dataset)
    dataset.extend(utils.load_subset(settings.CODEMIX_MIXED_FILE, codemix))


    # separately extract the text of the tweets for preprocessing.
    tweets = [d['text'] for d in dataset]

    # first, preprocess all those tweets that either require no/complete back-
    # transliteration.
    tweets[ : markers['translit-only-hin'] ] = utils.preprocess_tweets(
        tweets[ : markers['translit-only-hin'] ],
        tokenize=False,
        mask_user=True,
        mask_httpurl=True,
        mask_hashtag=False,
        mask_emoji=True
    )

    # strip user mentions, URLs, emojis, the leading # of hashtags
    for t in range(0, markers['translit-only-hin']):
        for symbol in [
            utils.USER_SYMBOL,
            utils.HTTPURL_SYMBOL,
            utils.EMOJI_SYMBOL,
            '#'
        ]:
            tweets[t] = tweets[t].replace(symbol, '')

    # back-transliterate those tweets which require complete back-
    # transliteration.
    for t in range(markers['translit-all'], markers['translit-only-hin']):
        tweets[t] = utils.back_transliterate(tweets[t])


    # next, handle the scenario where only some words have to be back-
    # transliterated.

    # the preprocessing remains the same, except that the tweets have to be
    # tokenized in order to back-transliterate only selective tokens.
    tweets[ markers['translit-only-hin'] : ] = utils.preprocess_tweets(
        tweets[ markers['translit-only-hin'] : ],
        tokenize=True,
        mask_user=True,
        mask_httpurl=True,
        mask_hashtag=False,
        mask_emoji=True
    )

    for t in range(markers['translit-only-hin'], len(tweets)):

        # strip the same things as before, except now they have to be removed
        # from a list of tokens instead of a single string.
        tweets[t] = [
            w.replace('#', '') for w in tweets[t]
            if w not in [
                utils.USER_SYMBOL, utils.HTTPURL_SYMBOL, utils.EMOJI_SYMBOL
            ]
        ]

        # adjust the language labels to match the tokens.
        dataset[t]['lang_labels'] = [
            dataset[t]['lang_labels'].get(w) for w in tweets[t]
        ]

        # back-transliterate only selective tokens.
        tweets[t] = utils.back_transliterate(
            tweets[t], lang_labels=dataset[t]['lang_labels']
        )

        # detokenize: join back all the tokens.
        tweets[t] = ' '.join(tweets[t])


    for t in range(len(tweets)):

        # tokenize into characters, add start- and end-of-sentence symbols
        tweets[t] = [utils.START_SYMBOL] + list(tweets[t]) + [utils.END_SYMBOL]

        # convert characters to indices
        tweets[t] = [char_to_idx.get(w) for w in tweets[t]]

        # filter out OOV characters (characters not in charset)
        tweets[t] = [w for w in tweets[t] if w is not None]

        # all preprocessing is done, put the tweets back in the dataset
        dataset[t]['text'] = tweets[t][:]


    # initialize the model
    model = CharLSTM(
        charset_size=len(charset)+1,    # +1 to accommodate for the padding token
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model.to(device)


    # training

    # prepare the dataset
    char_dataset = CharDataset(corpus)
    dataloader = DataLoader(
        char_dataset,
        batch_size=args.batch_size,
        shuffle=True,   # IMPORTANT! shuffle the corpus.
        pin_memory=True,
        collate_fn=char_dataset.collate_fn
    )

    # ignore_index=PADDING_TOKEN tells the loss function to ignore the padding
    # token in sequences while calculating the loss.
    loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=PADDING_TOKEN)

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
            input_seqs, seq_lengths, output_seqs = batch

            optimizer.zero_grad()

            pred_seqs = model(input_seqs, seq_lengths)

            # pred_seqs is shaped as batch-dim x sequence-dim x class-dim
            # loss_fn expects a tensor of shape batch-dim x class-dim x ...
            # therefore, we have to swap the axes with indices 1 and 2 (sequence
            # -dim and class-dim) to put pred_seqs in the required format.
            pred_seqs = pred_seqs.transpose(1, 2)

            loss = loss_fn(pred_seqs, output_seqs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % args.record_loss == args.record_loss-1:
                logging.info(
                    f'epoch {epoch_idx+1} batch {batch_idx+1}: loss = {running_loss:.6f}'
                )
                running_loss = 0.0


    # perplexity calculation

    # IMPORTANT! put the model in inference mode.
    model.eval()

    # prepare the dataset
    char_dataset = CharDataset([d['text'] for d in dataset])
    dataloader = DataLoader(
        char_dataset,
        batch_size=args.batch_size,
        shuffle=False,   # IMPORTANT! don't shuffle.
        pin_memory=True,
        collate_fn=char_dataset.collate_fn
    )

    perps = []

    with torch.no_grad():

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc='perplexity values',
            total=len(dataloader),
            unit='batch'
        ):

            batch = [b.to(device, non_blocking=True) for b in batch]
            input_seqs, seq_lengths, output_seqs = batch

            pred = model(input_seqs, seq_lengths)
            perps.extend(char_perplexity(pred, output_seqs, seq_lengths))


    # add these perplexities to the dataset
    for d in range(len(dataset)):
        dataset[d]['perplexity'] = perps[d]

    utils.write_perplexities(dataset, settings.PERPLEXITIES_FILE)


    # save the model
    torch.save(model.state_dict(), 'char_lstm.pt')
    print('Model saved to {}'.format('char_lstm.pt'))
