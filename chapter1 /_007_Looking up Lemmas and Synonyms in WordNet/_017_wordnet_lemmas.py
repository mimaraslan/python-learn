#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 02:34:44 2020

@author: mimaraslan
"""
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






