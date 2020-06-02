#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("------Collocations------")
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

words = [w.lower() for w in webtext.words('grail.txt')]
# print(words)
bcf = BigramCollocationFinder.from_words(words)
print(bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4))
# [("'", 's'), ('arthur', ':'), ('#', '1'), ("'", 't')]


print("------BigramAssocMeasures------")
from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset
bcf.apply_word_filter(filter_stops)
print(bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4))
# [('black', 'knight'), ('clop', 'clop'), ('head', 'knight'), ('mumble', 'mumble')]


print("------TrigramAssocMeasures------")
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures

words = [w.lower() for w in webtext.words('singles.txt')]
# print(words)
tcf = TrigramCollocationFinder.from_words(words)
tcf.apply_word_filter(filter_stops)
tcf.apply_freq_filter(3)
print(tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 4))
# [('long', 'term', 'relationship')]