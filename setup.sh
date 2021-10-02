#!/bin/bash

# this echoes the commands being run to the terminal, prefixed with a + sign,
# so that you can see what the script is doing.
set -x

models_root='models'
data_root='data'
mkdir "$models_root/"
mkdir "$data_root/"


# Step 1
# ------
# set +x here because the command need not be printed to the terminal.
# \033[A is a control character for moving the cursor up one line, it is
# required for deleting the previous set +x output.
( set +x && printf "\033[ASTEP 1 : Install Python packages
Please make sure you are running this script inside a virtualenv.\n\n" )

pip install -r 'requirements.txt'


# Step 2
# ------
# need some spaces following the \033[A character in order to erase the
# set +x output before printing a newline.
( set +x && printf "\033[A        ";
  printf "\nSTEP 2 : Downloading a clean English corpus (Brown corpus)\n\n" )

wget 'http://www.sls.hawaii.edu/bley-vroman/brown.txt'

mv 'brown.txt' "$data_root/"


# Step 3
# ------
( set +x && printf "\033[A        ";
  printf "\nSTEP 3 : Prepare lexicon from Brown corpus\n\n" )

# convert apostrophes ' to spaces, uppercase to lowercase
# replace spaces by newlines to put each word on a separate line
# remove punctuations, digits and \r characters from DOS line endings
# remove repeated occurrences of newlines
# sort and extract unique values
tr "'" ' ' < "$brown_root/brown.txt" | tr "[:upper:]" "[:lower:]" | tr ' ' "\n" | \
       tr -d "\r[:punct:][:digit:]" | tr -s "\n" | sort | uniq > "$data_root/lexicon.txt"


# Step 4
# ------
( set +x && printf "\033[A        ";
  printf "\nSTEP 4 : Downloading BERT tokenizer and model\n\n" )

mkdir "$models_root/bert/"

bert_weights_name='bert-base-uncased'

# Python code to stimulate download
python3 -c "from transformers import BertTokenizer, BertModel
BertTokenizer.from_pretrained('$bert_weights_name', cache_dir='$models_root/bert/')
BertModel.from_pretrained('$bert_weights_name', cache_dir='$models_root/bert/')
"


# Step 5
# ------
( set +x && printf "\033[A        ";
  printf "\nSTEP 5 : Downloading pretrained GloVe vectors
Both glove.6B as well as glove.twitter.27B (trained on Twitter data) are downloaded.\n\n" )

mkdir "$models_root/glove/"

wget 'https://nlp.stanford.edu/data/glove.6B.zip'
unzip 'glove.6B.zip'
mv glove.6B.* "$models_root/glove/"
rm 'glove.6B.zip'

wget 'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
unzip 'glove.twitter.27B.zip'
mv glove.twitter.27B.* "$models_root/glove/"
rm 'glove.twitter.27B.zip'


# Step 6
# ------
( set +x && printf "\033[A        ";
  printf "\nSTEP 6 : Downloading pretrained Fasttext vectors
Set of 1M vectors with subword information is downloaded.\n\n" )

mkdir "$models_root/fasttext/"

wget 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip'
unzip 'wiki-news-300d-1M-subword.vec.zip'

mv 'wiki-news-300d-1M-subword.vec' "$models_root/fasttext/" 
rm 'wiki-news-300d-1M-subword.vec.zip'


# Step 7
# ------
( set +x && printf "\033[A        ";
  printf "\nSTEP 7 : Downloading Sentiment140 dataset\n\n" )

# Python code to stimulate download
python3 -c "from datasets import load_dataset
load_dataset('sentiment140', cache_dir='$data_root/sentiment140/')
"

# Step 8
# ------
( set +x && printf "\033[A        ";
  printf "\nSTEP 8 : Modifying src/settings.py file\n\n" )

function modify_setting () {
	entry="$1"
	value="$2"

	# escape all forward slashes / in value
	value="${value//\//\\/}";

	sed -i "s/\(^$entry *= *\).*$/\1'$value'/" 'src/settings.py'
}

modify_setting 'BERT_ROOT' "$PWD/$models_root/bert/"
modify_setting 'GLOVE_ROOT' "$PWD/$models_root/glove/"
modify_setting 'FASTTEXT_ROOT' "$PWD/$models_root/fasttext/"
modify_setting 'SENTIMENT_ROOT' "$PWD/$data_root/sentiment140/"

( set +x && printf "\033[A        ";
  printf "\nALL DONE.\n" )
