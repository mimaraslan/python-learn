#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()
result = tokenizer.tokenize("Can't is a contraction.")
print (result)
#['Can', "'", 't', 'is', 'a', 'contraction', '.']