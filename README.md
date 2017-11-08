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

#### SVM English (18-10-2017)
Word n-grams (1,2) without Tweet Tokenizer: 0.3403<br />
Word n-grams (1,2) using Tweet Tokenizer: 0.4267<br />
Char n-grams (3,5) using Tweet Tokenizer: 0.4181<br />
Word n-grams (1,2) using Tweet Tokenizer & min_df = 10: 0.4148<br />
Word n-grams (1,2) using Tweet Tokenizer & min_df = 5: 0.4197<br />
Word n-grams (1,2) using Tweet Tokenizer & min_df = 2: 0.4255<br />
Word n-grams (1,3) using Tweet Tokenizer & min_df = default: 0.4267<br />
Word n-grams (1,2) using Tweet Tokenizer & min_df = default: 0.4210<br />
Word n-grams (1,3) using Tweet Tokenizer & min_df = default & stopwords: 0.4200<br />
Word n-grams (1,2) using Tweet Tokenizer & min_df = default & stopwords: 0.4180

#### SVM Spanish (25-10-2017)
Word n-grams (1,2) using Tweet Tokenizer & min_df = 4: 0.29640084686<br />
Word n-grams (1,3) using Tweet Tokenizer & min_df = default: 0.318681318681<br />
Char n-grams (1,3) using Tweet Tokenizer & min_df = default: 0.291057566287

#### Neural Network with word embeddings glove.twitter.27B (100D)
Neural Network (22-10-2017): 0.378

#### SVM FeatureUnion English (25-10-2017)
Word n-grams (1,3) combined with char n-grams (3,5) using Tweet Tokenizer & default min-df: 0.430482778618

#### SVM FeatureUnion Spanish (25-10-2017)
Word n-grams (1,3) combined with char n-grams (3,5) using Tweet Tokenizer & default min-df: 0.328359713681
20 iterations: 0.32967032967

### SVM Baseline with correct train/trial division
Word unigrams: 0.443 
Word n-grams(1,3) & char n-grams (3,5): 0.519 


