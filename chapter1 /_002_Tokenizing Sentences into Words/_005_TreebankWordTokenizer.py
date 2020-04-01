#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
english_mytext = tokenizer.tokenize('Hello World.')
print (english_mytext)
#['Hello', 'World', '.']