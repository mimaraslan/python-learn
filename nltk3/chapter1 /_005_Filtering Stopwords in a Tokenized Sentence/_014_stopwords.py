#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
english_stops = set(stopwords.words('english'))
words = ["Can't", 'is', 'a', 'contraction']
result = [word for word in words if word not in english_stops]
print (result)
#["Can't", 'contraction']