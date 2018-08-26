#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 18:55:15 2018

@author: k26609
"""

# https://www.datascience.com/resources/notebooks/word-embeddings-in-python 
# dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews

# imports
%matplotlib inline

import os
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import string
import re
from gensim import corpora
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#from ds_voc.text_processing import TextProcessing

# dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews
# sample for speed
file = r'/Users/k26609/Documents/GitHub/DeepLearning/WordEmbedding/Reviews.csv.zip'
raw_df = pd.read_csv(file, compression='zip', header=0, sep=',', quotechar='"')
raw_df = raw_df.sample(frac=0.1,  replace=False)
print(raw_df.shape) 

# grab review text
raw = list(raw_df['Text'])
print(len(raw))

#https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
table = str.maketrans({key: None for key in string.punctuation})
#new_s = s.translate(table)
def default_clean(inputString):
    return inputString.translate(table).lower()

stemmer = PorterStemmer() # from nltk.stem.porter 
stop_words = set(stopwords.words('english')) 
def stop_and_stem(inputString): 
    tokenized = [w for w in nltk.word_tokenize(inputString) if w not in stop_words]
    return [stemmer.stem(w) for w in tokenized]

"""
# word2vec expexts a list of list: each document is a list of tokens
te = TextProcessing()
cleaned = [te.default_clean(d) for d in raw]
sentences = [te.stop_and_stem(c) for c in cleaned]
"""
cleaned = [default_clean(d) for d in raw]
sentences = [stop_and_stem(c) for c in cleaned]

from gensim.models import Word2Vec

model = Word2Vec(sentences=sentences, # tokenized senteces, list of list of strings
                 size=300,  # size of embedding vectors
                 workers=4, # how many threads?
                 min_count=20, # minimum frequency per token, filtering rare words
                 sample=0.05, # weight of downsampling common words
                 sg = 0, # should we use skip-gram? if 0, then cbow
                 iter=5,
                 hs = 0
        )

X = model[model.wv.vocab]

print(model.wv.most_similar('peanut'))

print(model.wv.most_similar('coffee'))


# visualize food data
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.rcParams['figure.figsize'] = [10, 10]
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()




from bokeh.plotting import figure, show
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource, LabelSet

def interactive_tsne(text_labels, tsne_array):
    '''makes an interactive scatter plot with text labels for each point'''

    # define a dataframe to be used by bokeh context
    bokeh_df = pd.DataFrame(tsne_array, text_labels, columns=['x','y'])
    bokeh_df['text_labels'] = bokeh_df.index

    # interactive controls to include to the plot
    TOOLS="hover, zoom_in, zoom_out, box_zoom, undo, redo, reset, box_select"

    p = figure(tools=TOOLS, plot_width=700, plot_height=700)

    # define data source for the plot
    source = ColumnDataSource(bokeh_df)

    # scatter plot
    p.scatter('x', 'y', source=source, fill_alpha=0.6,
              fill_color="#8724B5",
              line_color=None)

    # text labels
    labels = LabelSet(x='x', y='y', text='text_labels', y_offset=8,
                      text_font_size="8pt", text_color="#555555",
                      source=source, text_align='center')

    p.add_layout(labels)

    # show plot inline
    output_notebook()
    show(p)

interactive_tsne(model.wv.vocab.keys(), X_tsne) # output to see in the notebook 

#

sent_w_pos = [nltk.pos_tag(d) for d in sentences]
sents = [[tup[0]+tup[1] for tup in d] for d in sent_w_pos]

model_pos = Word2Vec(sentences=sents,
                 size=300,
                 workers=4,
                 min_count=20,
                 sample=0.05,
                 sg = 0,
                 hs=0,
                 iter=5
        )


X = model_pos[model_pos.wv.vocab]


#

bigrams = Phrases(sentences)

model = Word2Vec(sentences=bigrams[sentences],
                 size=300,
                 workers=4,
                 min_count=20,
                 sample=0.05,
                 sg = 0,
                 iter=5,
                 hs = 0
        )

X = model[model.wv.vocab]
