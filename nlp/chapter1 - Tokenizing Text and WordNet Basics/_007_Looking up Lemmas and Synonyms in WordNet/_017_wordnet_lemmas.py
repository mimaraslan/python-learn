#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import wordnet

syn = wordnet.synsets('woman')[0]
lemmas = syn.lemmas()

print("--------------------------------------------")

print(len(lemmas))
# 2

print("--------------------------------------------")

print(lemmas[0].name())
# woman

print("--------------------------------------------")

print(lemmas[1].name())
# adult_female

print("--------------------------------------------")

print(lemmas[0].synset()) 
# Synset('woman.n.01')

print(lemmas[1].synset()) 
# Synset('woman.n.01')

print(lemmas[0].synset() == lemmas[1].synset())
# True

print("--------------------------------------------")

print([lemma.name() for lemma in syn.lemmas()])
# ['woman', 'adult_female']

print("--------------------------------------------")

synonyms = []

for syn in wordnet.synsets('woman'):
     for lemma in syn.lemmas():
         synonyms.append(lemma.name())         

print(len(synonyms)) 
# 11

print(synonyms) 
# ['woman', 'adult_female', 'woman', 'charwoman', 'char', 'cleaning_woman', 'cleaning_lady', 'woman', 'womanhood', 'woman', 'fair_sex']

print("--------------------------------------------")

print(len(set(synonyms)))
# 8

print(set(synonyms))

print("--------------------------------------------")

goodn2 = wordnet.synset('good.n.02')
print(goodn2.definition())
# a female person who plays a significant role (wife or mistress or girlfriend) in the life of a particular man

print("--------------------------------------------")

myResult = goodn2.lemmas()[0].antonyms()[0]
print(myResult.name())
# evil

print(myResult.synset().definition())
# the quality of being morally wrong in principle or practice

print("--------------------------------------------")

gooda1 = wordnet.synset('good.a.01')
print(gooda1.definition())
# 'having desirable or positive qualities especially those suitable for a thing specified'

myResult = gooda1.lemmas()[0].antonyms()[0]
print(myResult.name())
# 'bad'

print(myResult.synset().definition())
# 'having undesirable or negative qualities'







