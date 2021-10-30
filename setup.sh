#!/bin/bash


# ----------------------------------------------------------------------------
# This is a set-up script for the project.
# It downloads/installs all the required Python packages, datasets, pretrained
# models, and other utilities.
# Simply run it as: $ ./setup.sh
# For more information run: $ ./setup.sh -h
# ----------------------------------------------------------------------------


# modify entries in src/settings.py
function modify_setting () {
	entry="$1"
	value="$2"

	# escape all forward slashes / in value
	value="${value//\//\\/}";

	# replace the original entry with the new one
	sed -i "s/\(^$entry *= *\).*$/\1$value/" 'src/settings.py'
}


function pip_install () {
	cat <<EOF
---------------------------------------------------------------------
STEP 1 : Install Python packages
---------------------------------------------------------------------
EOF

	pip3 install -r 'requirements.txt'
}


function bert_download () {
	cat <<EOF
-------------------------------------------------------------------------
STEP 2 : Downloading BERT and co. (BERT, BERTweet, hindi-bert, IndicBERT)

Note that some Python packages and other libraries will be additionally
installed.
-------------------------------------------------------------------------
EOF

	# BERT
	bert_weights_name='bert-base-cased'
	bert_dirname='bert'
	mkdir "$models_root/$bert_dirname/"

	# Python code to stimulate download
	python3 -c "from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('$bert_weights_name', cache_dir='$models_root/$bert_dirname/')
AutoModel.from_pretrained('$bert_weights_name', cache_dir='$models_root/$bert_dirname/')
"

	# BERTweet
	bertweet_weights_name='vinai/bertweet-base'
	bertweet_dirname='bertweet'
	mkdir "$models_root/$bertweet_dirname/"

	python3 -c "from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('$bertweet_weights_name', cache_dir='$models_root/$bertweet_dirname/')
AutoModel.from_pretrained('$bertweet_weights_name', cache_dir='$models_root/$bertweet_dirname/')
"

	# hindi-bert
	hindibert_weights_name='monsoon-nlp/hindi-bert'
	hindibert_dirname='hindi-bert'
	mkdir "$models_root/$hindibert_dirname/"

	python3 -c "from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('$hindibert_weights_name', cache_dir='$models_root/$hindibert_dirname/')
AutoModel.from_pretrained('$hindibert_weights_name', cache_dir='$models_root/$hindibert_dirname/')
"

	# IndicBERT
	indicbert_weights_name='ai4bharat/indic-bert'
	indicbert_dirname='indic-bert'

	# IndicBERT requires sentencepiece package
	pip3 install 'sentencepiece'

	# IndicBERT requires protobuf
	pip3 install 'protobuf'
	wget 'https://github.com/protocolbuffers/protobuf/releases/download/v3.18.1/protoc-3.18.1-linux-x86_64.zip'
	unzip 'protoc-3.18.1-linux-x86_64.zip'

	# HACK: moving protoc binary to virtualenv
	# as long as virtualenv is activated it will be on path.
	mv 'bin/protoc' 'env/bin/'

	# remove other unnecessary files
	rm 'readme.txt' 'protoc-3.18.1-linux-x86_64.zip'
	rm -r 'include/'
	rmdir 'bin/'

	# actual IndicBERT download
	mkdir "$models_root/$indicbert_dirname/"
	python3 -c "from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('$indicbert_weights_name', cache_dir='$models_root/$indicbert_dirname/')
AutoModel.from_pretrained('$indicbert_weights_name', cache_dir='$models_root/$indicbert_dirname/')
"

	# $PWD gives the path of the current working directory;
	# this ensures that the full absolute path is written, instead of a
	# relative path.
	# single quotes are added to write the paths as Python strings.
	modify_setting 'BERT_ROOT' "'$PWD/$models_root/$bert_dirname/'"
	modify_setting 'BERTWEET_ROOT' "'$PWD/$models_root/$bertweet_dirname/'"
	modify_setting 'HINDIBERT_ROOT' "'$PWD/$models_root/$hindibert_dirname/'"
	modify_setting 'INDICBERT_ROOT' "'$PWD/$models_root/$indicbert_dirname/'"
}


function glove_download () {
	cat <<EOF
----------------------------------------------------------
STEP 3 : Downloading pretrained GloVe vectors (glove.840B)
----------------------------------------------------------
EOF
	glove_dirname='glove'

	# skip if already downloaded
	if [ -f "$models_root/$glove_dirname/glove.840B.300d.txt" ]; then
		return
	fi

	mkdir "$models_root/$glove_dirname/"
	wget 'https://nlp.stanford.edu/data/glove.840B.300d.zip'
	unzip 'glove.840B.300d.zip'
	mv 'glove.840B.300d.txt' "$models_root/$glove_dirname/"
	rm 'glove.840B.300d.zip'
	modify_setting 'GLOVE_ROOT' "'$PWD/$models_root/$glove_dirname/'"
}


function fasttext_download () {
	cat <<EOF
------------------------------------------------------------------------
STEP 4 : Downloading pretrained fastText vectors (crawl-300d-2M-subword)
------------------------------------------------------------------------
EOF
	fasttext_dirname='fasttext'

	# skip if already downloaded
	if [ -f "$models_root/$fasttext_dirname/crawl-300d-2M-subword.bin" ]; then
		return
	fi

	mkdir "$models_root/$fasttext_dirname/"
	wget 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
	unzip 'crawl-300d-2M-subword.zip'
	mv 'crawl-300d-2M-subword.bin' "$models_root/$fasttext_dirname/"
	rm 'crawl-300d-2M-subword.zip' 'crawl-300d-2M-subword.vec'
	modify_setting 'FASTTEXT_ROOT' "'$PWD/$models_root/$fasttext_dirname/'"
}


function indicft_download () {
	cat <<EOF
--------------------------------------------------------
STEP 5 : Downloading pretrained IndicFT model (lang: hi)
--------------------------------------------------------
EOF
	indicft_dirname='indicft'

	# skip if already downloaded
	if [ -f "$models_root/$indicft_dirname/indicnlp.ft.hi.300.bin" ]; then
		return
	fi

	mkdir "$models_root/$indicft_dirname/"
	wget 'https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/embedding-v2/indicnlp.ft.hi.300.bin'
	mv 'indicnlp.ft.hi.300.bin' "$models_root/$indicft_dirname/"
	modify_setting 'INDICFT_ROOT' "'$PWD/$models_root/$indicft_dirname/'"
}


function semeval17_download () {
	cat <<EOF
------------------------------------------------------------------------------
STEP 6 : Downloading SemEval-2017 Task 4 Subtask A dataset

NOTE : For the training data, this step only downloads the IDs and annotations
of the tweets. For downloading the actual text of the tweets, SemEval provides
a script which is also downloaded in this step. The script is run in a later
step and at that time the text of the tweets are scraped.
------------------------------------------------------------------------------
EOF

	semeval17_dirname='semeval-2017-4a'
	mkdir "$data_root/$semeval17_dirname/"

	# training data
	wget 'http://alt.qcri.org/semeval2017/task4/data/uploads/download.zip'
	unzip 'download.zip'

	# extract data corresponding only to subtask A, retain the README as well
	cat DOWNLOAD/Subtask_A/*.txt > "$data_root/$semeval17_dirname/train-dev-ids.txt"
	mv 'DOWNLOAD/README.txt' "$data_root/$semeval17_dirname/"
	rm -r '__MACOSX/'
	rm -r 'DOWNLOAD/'
	rm 'download.zip'

	# test data
	wget 'http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4-test.zip'
	unzip 'semeval2017-task4-test.zip'

	# again, extract test data corresponding only to subtask A
	mv 'SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt' \
		"$data_root/$semeval17_dirname/test.txt"
	rm -r '__MACOSX/'
	rm -r 'SemEval2017-task4-test/'
	rm 'semeval2017-task4-test.zip'

	# official tweet scraping script with some small modifications
	wget -nc 'https://gist.githubusercontent.com/saniya-m/e8f7704e1384a95b2cfc73772f62eefe/raw/e9cf72dbc43d5ec524d77e7d5f42e6f45e780909/download_tweets_api.py'
	modify_setting 'EN_TWEETS_ROOT' "'$PWD/$data_root/$semeval17_dirname/'"
}


function semeval20_download () {
	cat <<EOF
---------------------------------------------------------------------------
STEP 7 : Downloading SemEval-2020 Task 9 Dataset
NOTE : This dataset contains the text of all tweets so no separate scraping
step is involved.
---------------------------------------------------------------------------
EOF

	semeval20_dirname='semeval-2020-9'
	mkdir "$data_root/$semeval20_dirname/"

	wget -O 'Semeval_2020_task9_data.zip' \
		'https://zenodo.org/record/3974927/files/Semeval_2020_task9_data.zip?download=1'
	unzip 'Semeval_2020_task9_data.zip'

	# only extract the Hindi-English code-mixed data (not the Spanish-English one)
	# along with the README.
	mv 'Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt' \
		"$data_root/$semeval20_dirname/train.txt"
	mv 'Semeval_2020_task9_data/Hinglish/Hinglish_dev_3k_split_conll.txt' \
		"$data_root/$semeval20_dirname/dev.txt"
	mv 'Semeval_2020_task9_data/Hinglish/Hinglish_test_unalbelled_conll_updated.txt' \
		"$data_root/$semeval20_dirname/test.txt"
	mv 'Semeval_2020_task9_data/Hinglish/Hinglish_test_labels.txt' \
		"$data_root/$semeval20_dirname/test-labels.txt"
	mv 'Semeval_2020_task9_data/README.txt' "$data_root/$semeval20_dirname/"
	rm -r 'Semeval_2020_task9_data/'

	rm 'Semeval_2020_task9_data.zip'

	modify_setting 'HI_EN_TWEETS_ROOT' "'$PWD/$data_root/$semeval20_dirname/'"
}

function statmt_download () {
	cat <<EOF
-------------------------------------------------------
STEP 8 : Downloading statmt.org's 1 Billion Word Corpus
-------------------------------------------------------
EOF

	statmt_dirname='statmt-1bw'

	if [ -f "$data_root/$statmt_dirname/corpus.txt" ]; then
		return
	fi

	mkdir "$data_root/$statmt_dirname/"

	wget 'http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz'
	tar -xvzf '1-billion-word-language-modeling-benchmark-r13output.tar.gz'

	# this corpus comes in 'training' and 'heldout' splits.
	# in our case, there is no need for such a split, so we combine both
	# of them.
	cat 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* \
		1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/* > "$data_root/$statmt_dirname/corpus.txt"

	# taking a subset consisting of the first 1M lines of this corpus
	head -n '1000000' "$data_root/$statmt_dirname/corpus.txt" > 'temp.txt'
	mv 'temp.txt' "$data_root/$statmt_dirname/corpus.txt"

	rm -r '1-billion-word-language-modeling-benchmark-r13output/'
	rm '1-billion-word-language-modeling-benchmark-r13output.tar.gz'

	modify_setting 'EN_CORPUS_ROOT' "'$PWD/$data_root/$statmt_dirname/'"
}

function indiccorp_hi_download () {
	cat <<EOF
----------------------------------------------------------------
STEP 9 : Downloading IndicNLPSuite's Hindi Corpus (IndicCorp hi)
----------------------------------------------------------------
EOF

	indiccorp_hi_dirname='indiccorp-hi'

	if [ -f "$data_root/$indiccorp_hi_dirname/corpus.txt" ]; then
		return
	fi

	mkdir "$data_root/$indiccorp_hi_dirname/"

	wget 'https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indiccorp/hi.tar.xz'
	tar -xvf 'hi.tar.xz'
	mv 'data/hi/hi.txt' "$data_root/$indiccorp_hi_dirname/corpus.txt"

	rm 'hi.tar.xz'
	rmdir 'data/hi/'
	rmdir 'data/'

	# taking a subset consisting of the first 1M lines of this corpus
	head -n '1000000' "$data_root/$indiccorp_hi_dirname/corpus.txt" > 'temp.txt'
	mv 'temp.txt' "$data_root/$indiccorp_hi_dirname/corpus.txt"

	modify_setting 'HI_CORPUS_ROOT' "'$PWD/$data_root/$indiccorp_hi_dirname/'"
}

function tools_download () {
	cat <<EOF
---------------------------------------------------------------------
STEP 10 : Downloading external tools

* BERTweet's tweet normalization module
* Bhat et. al. (2014)'s transliteration tool

Note that some Python packages will be installed.
---------------------------------------------------------------------
EOF

	wget 'https://raw.githubusercontent.com/VinAIResearch/BERTweet/master/TweetNormalizer.py'
	pip3 install 'emoji'
	mv 'TweetNormalizer.py' 'src/'

	# skip installation of indic-trans if already installed
	python3 -c 'import indictrans' 2> /dev/null
	if [[ "$?" == 0 ]]; then
		return
	fi

	git clone 'https://github.com/libindic/indic-trans.git'
	cd 'indic-trans/'
	pip3 install -r 'requirements.txt'
	pip3 install .
	cd ..
	rm -rf 'indic-trans/'
}


function scrape_tweets () {
	cat <<EOF
-----------------------------------------------------------------------------
STEP 11 : Scraping tweets of SemEval-2017 Task 4 Subtask A

For this step, please make sure that you have a Twitter account, a Twitter
Developer Account and a registered app on the Twitter Developer Platform.
See https://developer.twitter.com/en/apply-for-access for more information.

You will be prompted to enter the API key and the API secret key of your app.
Then a browser window will open to authenticate you: it will ask you for your
Twitter username and password and will provide a PIN for you to enter back
into the terminal.

The scraping is done with the help of the Python package twitter, which is
installed in this step.

NOTE : The scraping takes a long time to complete.
You can stop it in the middle at any time by pressing CTRL-C.
To resume scraping, run this script again using
$ $0 -s
and it will start from where it was left off. You will be asked for your API
key and API secret key again, but the browser window will not open up.
-----------------------------------------------------------------------------
EOF

	semeval17_dirname='semeval-2017-4a'

	# requires the twitter Python package
	pip3 install 'twitter'

	python3 'download_tweets_api.py' --dist "$data_root/$semeval17_dirname/train-dev-ids.txt" --output "$data_root/$semeval17_dirname/train-dev.txt"

	# previous command might end in some connection/timeout error
	# in that case don't execute the following steps; execute only when
	# the scraping is successful.
	if (($? == 0)); then

		# now that we have the text of the tweets, we can remove the source file
		rm "$data_root/$semeval17_dirname/train-dev-ids.txt"

		# lot of tweets are no longer available, remove them
		grep -v 'Not Available' "$data_root/$semeval17_dirname/train-dev.txt" > 'temp.txt'
		mv 'temp.txt' "$data_root/$semeval17_dirname/train-dev.txt"

		# remove Twitter OAuth credentials stored here
		rm "$HOME/.my_app_credentials"

		# remove the scraping script, it is no longer required
		rm 'download_tweets_api.py'
	fi
}


function clean () {

	read -p "This will delete ALL of the data, models, Python packages and other tools (except the source code).\nAre you sure you want to proceed? (y/n) "

	if [[ "$REPLY" == 'n' ]]; then
		return
	fi

	rm -r "$data_root/"
	rm -r "$models_root/"

	rm -r 'env/'

	rm 'src/TweetNormalizer.py'

	# restore settings
	modify_setting 'BERT_ROOT' 'None'
	modify_setting 'BERTWEET_ROOT' 'None'
	modify_setting 'HINDIBERT_ROOT' 'None'
	modify_setting 'INDICBERT_ROOT' 'None'
	modify_setting 'GLOVE_ROOT' 'None'
	modify_setting 'FASTTEXT_ROOT' 'None'
	modify_setting 'INDICFT_ROOT' 'None'
	modify_setting 'EN_TWEETS_ROOT' 'None'
	modify_setting 'HI_EN_TWEETS_ROOT' 'None'
	modify_setting 'EN_CORPUS_ROOT' 'None'
	modify_setting 'HI_CORPUS_ROOT' 'None'
	modify_setting 'CHALLENGE_ROOT' 'None'
}


function show_help () {
	cat <<EOF
Usage: $0 [FLAGS]
Sets up environment, and downloads required models and datasets.
Options:
	-h	display this help and exit
	-c	clean environment, i.e. remove data, models, Python packages
		and other tools
	-s	scrape text of SemEval-2017 tweets

Ensure standard utilities like wget, unzip, tar and git are installed on
your system.

Create a virtual environment named env/ inside the current directory and
activate it, prior to running this script:
	$ python3 -m venv env
	$ source env/bin/activate

In case anything goes wrong and the script ends in error, you can rerun the
script:
	$ $0
It will not repeat downloads that take a long time and have completed
successfully.

The SemEval-2017 tweet scraping step may need to be repeated multiple times,
use
	$ $0 -s
to perform that single step only.
EOF
}


# set up root directories for containing datasets and pretrained models
models_root='models'
data_root='data'
mkdir -p "$models_root/"
mkdir -p "$data_root/"

# parse options
while getopts 'hcs' opt; do
	case "$opt" in
		h)
			show_help; exit 0;;
		c)
			clean; exit 0;;
		s)
			scrape_tweets; exit 0;;
		*)
			show_help >&2; exit 1;;
	esac
done
shift "$(($OPTIND-1))"


# execute all steps

# this is an additional step to set up a directory for storing the challenge
# test sets
challenge_root='challenge'
mkdir -p "$challenge_root/"
modify_setting 'CHALLENGE_ROOT' "'$PWD/$challenge_root/'"

pip_install
bert_download
glove_download
fasttext_download
indicft_download
semeval17_download
semeval20_download
statmt_download
indiccorp_hi_download
tools_download
scrape_tweets
