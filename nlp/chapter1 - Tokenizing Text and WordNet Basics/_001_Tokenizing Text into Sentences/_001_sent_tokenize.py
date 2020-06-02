#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import sent_tokenize
mytext = "Hello World. It's good to see you. Thanks for buying this book."
result = sent_tokenize(mytext)
print (result)
#['Hello World.', "It's good to see you.", 'Thanks for buying this book.']