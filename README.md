# pickleteam
Shared Task 2017

### Scattertext
The html files in */scattertext/html* can be opened in a web browser and contain plots showing the words most frequently occurring with one of the two emojis that are being compared.

In */scattertext/plot.py* change line 12 and 13 to the numbers corresponding with the emojis you want to compare. (See [CodaLab](https://competitions.codalab.org/competitions/17344) for the emojis and their numbers.)

To use *plot.py*, [Scattertext](https://github.com/JasonKessler/scattertext) must be installed.
On linux, run:

`$ pip install scattertext && python -m spacy.en.download`


### Results:
#### English
SVM (11-10-2017): 0.31

#### Spanish
SVM (11-10-2017): 0.23

#### With character unigrams
SVM (16-10-2017): 0.24

#### Using TweetTokenizer (unigrams)
SVM (16-10-2017): 0.41

#### MLP with TweetTokenizer and tf-idf word unigrams features
With parameters `min_df=10`, `hidden_layer_sizes=(50,)`, `max_iter=4`, `solver='adam'`: 0.43

#### SVM (18-10-2017)
Word n-grams (1,2) without Tweet Tokenizer: 0.3403
Word n-grams (1,2) using Tweet Tokenizer: 0.4267
Char n-grams (3,5) using Tweet Tokenizer: 0.4181
Word n-grams (1,2) using Tweet Tokenizer & min_df = 10: 0.4148
Word n-grams (1,2) using Tweet Tokenizer & min_df = 5: 0.4197
Word n-grams (1,2) using Tweet Tokenizer & min_df = 2: 0.4255
Word n-grams (1,3) using Tweet Tokenizer & min_df = default: 0.4267
Word n-grams (1,2) using Tweet Tokenizer & min_df = default: 0.4210
Word n-grams (1,3) using Tweet Tokenizer & min_df = default & stopwords: 0.4200
Word n-grams (1,2) using Tweet Tokenizer & min_df = default & stopwords: 0.4180

#### Neural Network with word embeddings glove.twitter.27B (100D)
Neural Network (22-10-2017): 0.378




