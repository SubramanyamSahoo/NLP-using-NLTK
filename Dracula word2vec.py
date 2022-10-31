# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:25:17 2022

@author: asus
"""


import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """The estate is called Carfax, no doubt a corruption of the old Quatre Face, as
the house is four-sided, agreeing with the cardinal points of the compass. It
contains in all some twenty acres, quite surrounded by the solid stone wall above
mentioned. There are many trees on it, which make it in places gloomy, and
there is a deep, dark-looking pond or small lake, evidently fed by some springs,
as the water is clear and flows away in a fair-sized stream. The house is very
large and of all periods back, I should say, to mediæval times, for one part is of
stone immensely thick, with only a few windows high up and heavily barred
with iron. It looks like part of a keep, and is close to an old chapel or church. I
could not enter it, as I had not the key of the door leading to it from the house,
but I have taken with my kodak views of it from various points. The house has
been added to, but in a very straggling way, and I can only guess at the amount
of ground it covers, which must be very great. There are but few houses close at
hand, one being a very large house only recently added to and formed into a
private lunatic asylum. It is not, however, visible from the grounds."""



# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.key_to_index 

# Finding Word Vectors
vector = model.wv['ground']

# Most similar words
similar = model.wv.most_similar('ground')