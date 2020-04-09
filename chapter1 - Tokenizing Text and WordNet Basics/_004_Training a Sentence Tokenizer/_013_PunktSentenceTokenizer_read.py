#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import PunktSentenceTokenizer

with open('mytext.txt', encoding='ISO-8859-2') as f:  text = f.read()
#with open('/Users/mimaraslan/nltk_data/corpora/webtext/overheard.txt', encoding='ISO-8859-2') as f:  text = f.read()
#with open('/usr/share/nltk_data/corpora/webtext/overheard.txt', encoding='ISO-8859-2') as f: text = f.read()
sent_tokenizer = PunktSentenceTokenizer(text)
sents = sent_tokenizer.tokenize(text)
print (sents[0])
#'White guy: So, do you have any plans for this evening?'
print (sents[2])
#'Girl: But you already have a Big Mac...'