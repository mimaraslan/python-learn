#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext

text = webtext.raw('overheard.txt')

sent_tokenizer = PunktSentenceTokenizer(text)
sents1 = sent_tokenizer.tokenize(text)
print (sents1[0])
# White guy: So, do you have any plans for this evening?

from nltk.tokenize import sent_tokenize
sents2 = sent_tokenize(text)
print (sents2[0])
#'White guy: So, do you have any plans for this evening?'


print (sents1[678])
#'Girl: But you already have a Big Mac...'

print (sents2[678])
#Girl: But you already have a Big Mac... Hobo: Oh, this is all theatrical.