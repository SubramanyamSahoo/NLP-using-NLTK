# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:02:10 2022

@author: asus
"""

import nltk

paragraph =  """The estate is called Carfax, no doubt a corruption of the old Quatre Face, as
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
               
               
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()