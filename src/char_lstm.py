"""Character-level LSTM trained on clean, canonical corpora,
and used to find the perplexity of noisy samples.
"""

import argparse
import logging
import os

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


# sanity checks for the model.
# two samples are provided each for English and Hindi.
# in both cases, the first sample is the first line of the canonical corpus
# and has been encountered by the model during training,
# while the second sample is some nonsensical text.
# intuitively, the perplexity of the first sample should be lower than that
# of the second.
# run this script in --mode 'debug' to verify the same.
# also verify that the difference in perplexity should increase with more
# training (more epochs).

EN_SANITY_CHECKS = (
    'The U.S. Centers for Disease Control and Prevention initially advised school systems to close if outbreaks occurred, then reversed itself, saying the apparent mildness of the virus meant most schools and day care centers should stay open, even if they had confirmed cases of swine flu.',
    'oiojdalkjdslkasajdlka'
)
HI_SANITY_CHECKS = (
    u'आवेदन करने की आखिरी तारीख 31 जनवरी, 2020 है।',
    u'कखाीूैपूैतंिपुफाडडड'
)


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

    def __init__(self, charset_size, hidden_size, bidirectional=False):
        """Initialize the model.

        Pass the size of character set (charset), the hidden size of the LSTM,
        and whether the LSTM should be bidirectional or not.
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
            num_layers=1,
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



def filter_tweet(text):
    text = text.replace(utils.USER_SYMBOL, '')
    text = text.replace(utils.HTTPURL_SYMBOL, '')
    text = text.replace(utils.EMOJI_SYMBOL, '')
    text = text.replace('#', '')
    return text


if __name__ == '__main__':


    # command-line arguments
    # ----------------------

    MODE_DEBUG, MODE_DEV, MODE_FULL = 'debug', 'dev', 'full'

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--lang', choices=[utils.LANG_ENG, utils.LANG_HIN], default=utils.LANG_ENG,
        help='train English/Hindi LSTM'
    )
    parser.add_argument(
        '--mode', choices=[MODE_DEBUG, MODE_DEV, MODE_FULL], default=MODE_DEV,
        help='run this script in debug/dev/full mode'
    )
    parser.add_argument(
        '--hidden-size', type=int, default=50, help='size of LSTM hidden layer'
    )
    parser.add_argument(
        '--bidirectional', action='store_true', default=False,
        help='use a bidirectional LSTM'
    )
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
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


    # use smaller subsets of the whole corpus in debug/dev mode
    num_lines = None
    if args.mode == MODE_DEBUG:
        num_lines = 20
    elif args.mode == MODE_DEV:
        num_lines = 20000


    if args.lang == utils.LANG_ENG:

        # load the English corpus
        with open(os.path.join(settings.EN_CORPUS_ROOT, 'corpus.txt'), 'r') as f:

            if num_lines is None:
                corpus = f.readlines()      # read all lines
            else:

                # load only as many lines as required; this is faster and more
                # efficient than loading all the lines and keeping only a few
                # of them.
                corpus = [f.readline() for _ in range(num_lines)]

        # dataset of noisy samples whose perplexity values have to be calculated
        if args.mode == MODE_DEBUG:
            perp_dataset = EN_SANITY_CHECKS
        else:
            perp_dataset = utils.load_en_tweets_dataset(split='train')
            perp_dataset = perp_dataset + utils.load_en_tweets_dataset(split='dev')
            perp_dataset = perp_dataset + utils.load_en_tweets_dataset(split='test')

            perp_ids = [row['tweet_id'] for row in perp_dataset]
            perp_dataset = [row['text'] for row in perp_dataset]

    else:

        # load the Hindi corpus
        # note the utf-8 encoding for Devanagari characters
        with open(
            os.path.join(settings.HI_CORPUS_ROOT, 'corpus.txt'),
            'r',
            encoding='utf-8'
        ) as f:
            if num_lines is None:
                corpus = f.readlines()
            else:
                corpus = [f.readline() for _ in range(num_lines)]

        if args.mode == MODE_DEBUG:
            perp_dataset = HI_SANITY_CHECKS
        else:
            # TODO
            pass

    if args.lang == utils.LANG_ENG:

        # the English corpus is in a slightly different format (note the
        # spacing around punctuations); it needs to be detokenized as follows
        # to resemble normal English text.
        detok = TreebankWordDetokenizer()
        corpus = [detok.detokenize(doc.split(' ')) for doc in corpus]

    if args.mode != MODE_DEBUG:

        perp_dataset = utils.preprocess_tweets(
            perp_dataset,
            tokenize=False,
            mask_user=True,
            mask_httpurl=True,
            mask_hashtag=False,
            mask_emoji=True
        )
        perp_dataset = [filter_tweet(text) for text in perp_dataset]

        if args.lang == utils.LANG_HIN:
            perp_dataset = [
                utils.back_transliterate(text) for text in perp_dataset
            ]

    # add start- and end-of-sentence tags; tokenize into characters
    corpus = [
        [utils.START_SYMBOL] + list(doc) + [utils.END_SYMBOL]
        for doc in corpus
    ]
    perp_dataset = [
        [utils.START_SYMBOL] + list(doc) + [utils.END_SYMBOL]
        for doc in perp_dataset
    ]

    # the set of unique characters in the corpus (the "vocabulary" for a
    # character-level model)
    charset = sorted(set([c for doc in corpus for c in doc]))

    # convert characters to indices
    # adding 1 to the indices to accommodate for the padding token at
    # position 0.
    char_to_idx = {c: i+1 for i, c in enumerate(charset)}
    corpus = [[char_to_idx[c] for c in doc] for doc in corpus]


    # perform the same conversion on perp_dataset
    # however, perp_dataset may contain OOV characters (characters not in
    # charset) -- ignore these.

    perp_dataset = [
        [char_to_idx.get(c, None) for c in doc] for doc in perp_dataset
    ]

    # filter out OOV characters
    perp_dataset = [
        [c for c in doc if c is not None] for doc in perp_dataset
    ]

    # initialize the model
    # adding one to the size of the charset, again to accommodate for the
    # padding token.
    model = CharLSTM(
        charset_size=len(charset)+1,
        hidden_size=args.hidden_size,
        bidirectional=args.bidirectional
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model.to(device)


    # training
    # --------

    # prepare the dataset
    dataset = CharDataset(corpus)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    # cross entropy loss
    # ignore_index=PADDING_TOKEN tells the loss function to ignore the padding
    # token in sequences while calculating the loss.
    loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=PADDING_TOKEN)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # set up logging
    logging.basicConfig(filename='train.log', filemode='w', level=logging.INFO)
    print('Periodic record of training loss will be written to train.log.')

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
    # ----------------------

    # put the model in inference mode
    model.eval()

    # prepare the dataset
    dataset = CharDataset(perp_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    # perplexity values
    perps = []

    with torch.no_grad():

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc='perplexity calculation',
            total=len(dataloader),
            unit='batch'
        ):

            batch = [b.to(device, non_blocking=True) for b in batch]
            input_seqs, seq_lengths, output_seqs = batch

            pred = model(input_seqs, seq_lengths)
            perps.extend(char_perplexity(pred, output_seqs, seq_lengths))


    if args.mode == MODE_DEBUG:
        print('Perplexity of:')
        print('Sample clean text: {:.6f}'.format(perps[0]))
        print('Sample noisy text: {:.6f}'.format(perps[1]))
    else:

        # write the perplexity values to file
        filename = 'tweet_perplexities_{}.txt'.format(args.lang)
        with open(filename, 'w') as f:

            # write the tweet ID and perplexity value, separated by a comma
            f.write('\n'.join([
                '{},{:.6f}'.format(i, p) for i, p in zip(perp_ids, perps)
            ]))

        print('Perplexities written to {}'.format(filename))


    # save the model
    filename = 'char_lstm_{}.pt'.format(args.lang)
    torch.save(model.state_dict(), filename)
    print('Model saved to {}'.format(filename))
