#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import regexp_tokenize

result = regexp_tokenize("Can't @ is # ' a £ 1 ! 9  contraction.", "[\w']+")
print (result)
#["Can't", 'is', "'", 'a', '1', '9', 'contraction']

