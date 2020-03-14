#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('\s+', gaps=True)
result = tokenizer.tokenize("Can't @ is # ' a £ 1 ! 9  cont@ra'ct!ion.")
print (result)
#["Can't", '@', 'is', '#', "'", 'a', '£', '1', '!', '9', "cont@ra'ct!ion."]