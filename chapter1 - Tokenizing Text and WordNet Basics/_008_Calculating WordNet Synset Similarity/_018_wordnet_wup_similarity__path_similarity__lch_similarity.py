#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import wordnet
cb = wordnet.synset('cookbook.n.01')
ib = wordnet.synset('instruction_book.n.01')
print(cb.wup_similarity(ib)) # Wu-Palmer Similarity
# 0.9166666666666666

print("--------------------------------------------")

ref = cb.hypernyms()[0]
print(cb.shortest_path_distance(ref))
# 1

print(ib.shortest_path_distance(ref))
# 1

print(cb.shortest_path_distance(ib))
# 2

print("--------------------------------------------")

dog = wordnet.synsets('dog')[0]
print(dog.wup_similarity(cb))
# 0.38095238095238093

print(sorted(dog.common_hypernyms(cb)))
# [Synset('entity.n.01'), Synset('object.n.01'), Synset('physical_entity.n.01'), Synset('whole.n.02')]

print("------Comparing verbs------")

cook = wordnet.synset('cook.v.01')
print(cook.definition()+"\n") 

bake = wordnet.synset('bake.v.02')
print(bake.definition()+"\n") 

print(cook.wup_similarity(bake))
# 0.6666666666666666

print("------Path and Leacock Chordorow (LCH) similarity------")

print(cb.path_similarity(ib))
# 0.3333333333333333

print(cb.path_similarity(dog))
# 0.07142857142857142

print(cb.lch_similarity(ib))
# 2.538973871058276

print(cb.lch_similarity(dog))
# 0.9985288301111273