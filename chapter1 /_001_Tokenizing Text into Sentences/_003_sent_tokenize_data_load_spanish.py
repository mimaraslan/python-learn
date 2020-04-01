#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk.data

english_mytext = "Hello World. It's good to see you. Thanks for buying this book."
english_tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
english_result = english_tokenizer.tokenize(english_mytext)
print (english_result)
# ['Hello World.', "It's good to see you.", 'Thanks for buying this book.']

print ("----------------------------")

spanish_mytext = 'Hola amigo. Estoy bien.'
spanish_tokenizer = nltk.data.load('tokenizers/punkt/PY3/spanish.pickle')
spanish_result = spanish_tokenizer.tokenize(spanish_mytext)
print (spanish_result)
# ['Hola amigo.', 'Estoy bien.']
