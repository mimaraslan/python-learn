#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer("[\w']+")
result = tokenizer.tokenize("Can't @ is # ' a £ 1 ! 9  contraction.")
print (result)
#["Can't", 'is', "'", 'a', '1', '9', 'contraction']