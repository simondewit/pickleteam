import spacy
import pickle
import pandas as pd

from scattertext import SampleCorpora
from scattertext import produce_scattertext_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas

def main():
    nlp     = spacy.en.English()

    emoji_1 = '11'  #--- choose emoji nr 0-19
    emoji_2 = '17'  #--- choose emoji nr 0-19

    tweets  = []
    with open('tweetsAsTuplesFile2.pickle','rb') as f:
        tweetsAsTuplesFile = pickle.load(f)
        for elem in tweetsAsTuplesFile:
            if elem[0] == emoji_1 or elem[0] == emoji_2:
                tweets.append([elem[0], elem[1]])

    tweets_df = pd.DataFrame(tweets, columns=['emoji', 'tweet'])
    corpus    = CorpusFromPandas(tweets_df, category_col='emoji', text_col='tweet', nlp=nlp).build()

    html = produce_scattertext_explorer(corpus,
                                        category               = emoji_1,
                                        category_name          = emoji_1,
                                        not_category_name      = emoji_2,
                                        minimum_term_frequency = 5,
                                        width_in_pixels        = 1000)

    open('html/emoji_'+emoji_1+'_vs_'+emoji_2+'.html', 'wb').write(html.encode('utf-8'))
    print('Open html/emoji_'+emoji_1+'_vs_'+emoji_2+'.html in Chrome or Firefox.')

if __name__ == '__main__':
    main()