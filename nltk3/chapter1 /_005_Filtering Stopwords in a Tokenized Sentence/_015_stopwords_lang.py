#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import stopwords

print (stopwords.fileids())
#['arabic', 'azerbaijani', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek', 'hungarian', 'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene', 'spanish', 'swedish', 'tajik', 'turkish']

print ("-------------------------------------------------")

print (stopwords.words('english'))

print ("-------------------------------------------------")

print (stopwords.words('turkish'))