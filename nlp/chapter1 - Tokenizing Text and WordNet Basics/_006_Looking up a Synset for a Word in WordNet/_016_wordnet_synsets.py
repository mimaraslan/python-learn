#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus import wordnet

syn = wordnet.synsets('woman')[0]

print("--------------------------------------------")

print(syn.name())
#'woman.n.01'

print("--------------------------------------------")

print(syn.definition())
#'an adult female person (as opposed to a man)'

print("--------------------------------------------")

print(wordnet.synset('woman.n.01'))
# Synset('woman.n.01')

print("--------------------------------------------")

print(wordnet.synsets('woman')[0].examples())
# ['the woman kept house while the man hunted']

print("--------------------------------------------")

print(syn.hypernyms())
# [Synset('adult.n.01'), Synset('female.n.02')]

print("--------------------------------------------")

print(syn.hypernyms()[0].hyponyms())
# [Synset('brachycephalic.n.01'), Synset('caregiver.n.02'), Synset('catch.n.03'), Synset('centrist.n.01'), Synset('character.n.05'), Synset('conservative.n.01'), Synset('dolichocephalic.n.01'), Synset('elder.n.01'), Synset('ex-spouse.n.01'), Synset('host.n.01'), Synset('important_person.n.01'), Synset('jack_of_all_trades.n.01'), Synset('liberal.n.01'), Synset('liberal.n.02'), Synset('man.n.01'), Synset('militarist.n.01'), Synset('oldster.n.01'), Synset('pacifist.n.01'), Synset('patrician.n.01'), Synset('pledgee.n.01'), Synset('pledger.n.01'), Synset('professional.n.01'), Synset('sobersides.n.01'), Synset('sophisticate.n.01'), Synset('stay-at-home.n.01'), Synset('stoic.n.02'), Synset('thoroughbred.n.01'), Synset('woman.n.01')]

print("--------------------------------------------")

print(syn.root_hypernyms())
# [Synset('entity.n.01')]

print("--------------------------------------------")

print(syn.hypernym_paths())
# [[Synset('entity.n.01'), Synset('physical_entity.n.01'), Synset('object.n.01'), Synset('whole.n.02'), Synset('artifact.n.01'), Synset('creation.n.02'), Synset('product.n.02'), Synset('work.n.02'), Synset('publication.n.01'), Synset('book.n.01'), Synset('reference_book.n.01'), Synset('cookbook.n.01')]]

print("--------------------------------------------")

print(syn.pos())
# 'n'

print("--------------------------------------------")

print(wordnet.synsets('woman'))

for i in range(len(wordnet.synsets('woman'))):
  print(wordnet.synsets('woman')[i].definition()+"\n") 
else:
  print("Finally finished!")
  
# an adult female person (as opposed to a man)
# a female person who plays a significant role (wife or mistress or girlfriend) in the life of a particular man
# a human female employed to do housework
# women as a class  
  
# print(wordnet.synsets('woman')[0].definition()+"\n") 
# print(wordnet.synsets('woman')[1].definition()+"\n")
# print(wordnet.synsets('woman')[2].definition()+"\n")
# print(wordnet.synsets('woman')[3].definition()+"\n")

print(len(wordnet.synsets('woman')))
# 4

print("--------------------------------------------")

# Part of speech Tag
# Noun  n 
# Adjective  a 
# Adverb  r 
# Verb  v

print(len(wordnet.synsets('woman', pos='n')))
# 4

print("--------------------------------------------")

print(len(wordnet.synsets('woman', pos='a')))
# 0
