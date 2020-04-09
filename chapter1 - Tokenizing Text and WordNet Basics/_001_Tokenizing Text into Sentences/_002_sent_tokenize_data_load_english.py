#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk.data
mytext = "Hello World. It's good to see you. Thanks for buying this book."
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
result = tokenizer.tokenize(mytext)
print (result)
#['Hello World.', "It's good to see you.", 'Thanks for buying this book.']