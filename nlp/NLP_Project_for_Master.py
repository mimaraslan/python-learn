# -*- coding: utf-8 -*-
"""Mimar Aslan, NLP_Project_for_Master
    
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    !pip install autocorrect
    !pip install spellchecker
    !pip install pyspellchecker
    !pip install Keras
    !pip install plotly
    !pip install plotly --upgrade
"""
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import cufflinks
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from autocorrect import Speller
from spellchecker import SpellChecker
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from bs4 import BeautifulSoup
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import plotly.figure_factory as ff
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

print("En güncel İngilizce stopwords paketlerini projemize çekiyoruz.")
nltk.download('stopwords')

pd.options.display.max_colwidth = 9999999999999
WPT = nltk.WordPunctTokenizer()
stopWordList = nltk.corpus.stopwords.words('english')
spell = Speller(lang='en')

class MyRepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word
        replaceWord = self.repeat_regexp.sub(self.repl, word)

        if replaceWord != word:
            return self.replace(replaceWord)
        else:
            return replaceWord


replacer = MyRepeatReplacer()


def uniquify(string):
    output = []
    seen = set()
    for word in string.split():
        if word not in seen:
            output.append(word)
            seen.add(word)
    return ' '.join(output)


def readFiles(mypath):
    print("Verilen klasörü tara ve txt uzantılı dosyları al, oku ve üst klasörlerini de label olarak kullan.")
    basicList = []
    labelList = []
    document_list = []
    for path, subdirs, files in os.walk(mypath):
        for name in files:
            if os.path.splitext(os.path.join(path, name))[1] == ".txt":
                document_list.append(os.path.join(path, name))
    #print(document_list)
    for pathList in document_list:
        folder = os.path.basename(os.path.dirname(pathList))
        label = folder
        #print(label)

        dataFile = open(pathList, 'r',  errors='ignore')
        data = dataFile.read()
       
        # Kelimeleri ayır
        singleDoc = re.sub(" \d+", " ", data)
        pattern = r"[{}]".format(",.;")
        singleDoc = re.sub(pattern, "", singleDoc)
        
        # Dokümanı küçük harflere çevir
        singleDoc = singleDoc.lower()
        singleDoc = singleDoc.strip()
        tokens = WPT.tokenize(data)
        
        # Stop-word listesindeki kelimeler hariç digerlerini al
        filteredTokens = [token for token in tokens if token not in stopWordList]
        singleDoc = ' '.join(filteredTokens)
        MY_TEXT = singleDoc
        basicList.append(MY_TEXT)
        labelList.append(label)
    return basicList, labelList


def norm_doc(singleDoc):
    # Remove special characters and numbers
    # Dokümandan belirlenen özel karakterleri ve sayıları at
    singleDoc = re.sub(" \d+", " ", singleDoc)
    pattern = r"[{}]".format(",.;")
    singleDoc = re.sub(pattern, "", singleDoc)
    
    # Dokümanı küçük harflere çevir
    singleDoc = singleDoc.lower()
    singleDoc = singleDoc.strip()

    # Dokümanı token'larına ayır
    tokens = WPT.tokenize(singleDoc)

    # Listeye eş anlamlıları da ekle
    synonyms = []
    for myToken in tokens:
        for syn in wordnet.synsets(myToken):
            for lm in syn.lemmas():
                synonyms.append(lm.name())  

    # Stop-word listesindeki kelimeler hariç digerlerini al
    filteredTokens = []
    for myToken in synonyms:
        if myToken not in stopWordList:
            filteredTokens.append(myToken)

    # Tekrar eden karakterleri sil ve yanlış yazılmış kelime düzelt
    unrepeated = []
    for i in filteredTokens:
        noLetter = replacer.replace((i))
        unrepeated.append(spell(noLetter))


    # Dokümanı tekrar oluştur
    singleDoc = ' '.join(unrepeated)
  
    # Tekrar eden kelimeleri siliyorum
    singleDoc = uniquify(singleDoc)
    return singleDoc


print("\nBütün dosyaları okumaya basla.")
docs, classes = np.array(readFiles("/content/drive/My Drive/Master/NLP/DataSet"))

docs = np.array(docs)
print("Ön işlemleri yap.")
normDocs = np.vectorize(norm_doc)
normalizedDocuments = normDocs(docs)

dfDocs = pd.DataFrame({'Icerik': normalizedDocuments,
                        'Kategori': classes})

    
data = dfDocs[['Icerik', 'Kategori']]

text = data['Icerik']
labels = data['Kategori']
print("Kategori")
print(labels)
print("-------------------------------------------------------------")

print("@title Default title text")
print("1.Terim Sayma Adımları")
print("Bag of words (BoW) Matrix çıkartılıyor")
bagOfWords = CountVectorizer(min_df=0.25)
bagOfMatrix = bagOfWords.fit_transform(normalizedDocuments)
print(bagOfMatrix)
print("-------------------------------------------------------------")


print("Bag of words (BoW) Vector içerisindeki tüm öznitelikleri (features) al")
features = bagOfWords.get_feature_names()

print("Bazı örnek Bag of words (BoW) Vector öznitelikleri (features) göster")
print("features[33]:" + features[33])
print("features[44]:" + features[44])
print("features[55]:" + features[55])
print("-------------------------------------------------------------")

bagOfMatrix = bagOfMatrix.toarray()
print(bagOfMatrix)
print("-------------------------------------------------------------")

print("Doküman öznitelik matrisini göster (document by term matrice)")
bagOfDf = pd.DataFrame(bagOfMatrix, columns=features)
print(bagOfDf)
print("-------------------------------------------------------------")

print("Doküman - info göster")
print(bagOfDf.info())
print("-------------------------------------------------------------")


# 2.TFxIdf Hesaplama Adımları
print("TF, TF-IDF, 1-gram Hesapla...")
tFidfVector = TfidfVectorizer(ngram_range=(1, 1), min_df=0.01)
tFidfMatrix = tFidfVector.fit_transform(normalizedDocuments)
tFidfMatrix = tFidfMatrix.toarray()
print(np.round(tFidfMatrix,2))
print("-------------------------------------------------------------")


print("tFidfVector içerisindeki tüm öznitelikleri al")
features = tFidfVector.get_feature_names()

print("Doküman - öznitelik matrisini göster")
tFidfDf = pd.DataFrame(np.round(tFidfMatrix, 2), columns=features)
print(tFidfDf)
print("-------------------------------------------------------------")

print("Laplace değer 0 ise 1 ekle")
tFidfDf[tFidfDf==0] += 1
print(tFidfDf)
print("-------------------------------------------------------------")


print("TF, TF-IDF, 2-gram Hesapla...")
tFidfVector = TfidfVectorizer(ngram_range=(1, 2), min_df=0.01)
tFidfMatrix = tFidfVector.fit_transform(normalizedDocuments)
tFidfMatrix = tFidfMatrix.toarray()
print(np.round(tFidfMatrix,2))
print("-------------------------------------------------------------")

print("tFidfVector içerisindeki tüm öznitelikleri al")
features = tFidfVector.get_feature_names()
print("Doküman - öznitelik matrisini göster")
tFidfDf = pd.DataFrame(np.round(tFidfMatrix, 2), columns=features)
print(tFidfDf)
print("-------------------------------------------------------------")

print("Laplace değer 0 ise 1 ekle")
tFidfDf[tFidfDf==0] += 1
print(tFidfDf)
print("-------------------------------------------------------------")

print("TF, TF-IDF, 3-gram Hesapla...")
tFidfVector = TfidfVectorizer(ngram_range=(1, 3), min_df=0.01)
tFidfMatrix = tFidfVector.fit_transform(normalizedDocuments)
tFidfMatrix = tFidfMatrix.toarray()
print(np.round(tFidfMatrix,2))
print("-------------------------------------------------------------")

print("tFidfVector içerisindeki tüm öznitelikleri al")
features = tFidfVector.get_feature_names()
print("Doküman - öznitelik matrisini göster")
tFidfDf = pd.DataFrame(np.round(tFidfMatrix, 2), columns=features)
print(tFidfDf)
print("-------------------------------------------------------------")

print("Laplace değer 0 ise 1 ekle")
tFidfDf[tFidfDf==0] += 1
print(tFidfDf)
print("-------------------------------------------------------------")
print('#'*60)
print("-------------------------------------------------------------")

######################################################################################

# MultinomialNB, GaussianNB ve BernoulliNB 3 tane farklı Naive Bayes Sınıfı vardır.
# Duruma göre bu üç sınıftan birini seçebiliriz. Yapılan seçim de modelin başarı durumunu etkiler.
# MultinomialNB : Tahmin edeceğiniz veri veya kolon nominal ise (Int sayılar )
# GaussianNB : Tahmin edeceğiniz veri veya kolon sürekli (real,ondalıklı vs.) ise
# BernoulliNB : Tahmin edeceğiniz veri veya kolon ikili ise (Evet/Hayır , Sigara içiyor/ İçmiyor vs.)

# MultinomialNB : Tahmin edeceğiniz veri veya kolon nominal ise ( Int sayılar )
print("MultinomialNB eğitime başla... 1-gram")
countVector = CountVectorizer(ngram_range=(1, 1), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)

print("MultinomialNB modeli çıkart... MultinomialNB sınıfından bir nesne ürettik.")
multinomialNB = MultinomialNB()

# print("Makineyi eğitiyoruz.)
multinomialNB = multinomialNB.fit(xTrain, yTrain)

# Test veri kümemizi verdik ve  tahmin etmesini sağladık
result = multinomialNB.predict(xTest.todense())

print("MultinomialNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)


print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score = f1_score(yTest, result, average="macro")
precisionScore = precision_score(yTest, result, average="macro")
recallScore = recall_score(yTest, result, average="macro")

print('MultinomialNB F1 Score {}'.format(f1Score))
print('MultinomialNB Precision Score {}'.format(precisionScore))
print('MultinomialNB Recall Score {}'.format(recallScore))
print('MultinomialNB Başarı Oranı {}'.format(accuracy))
print("-------------------------------------------------------------")      
######################################################################################


print("MultinomialNB eğitime başla... 2-gram")
countVector = CountVectorizer(ngram_range=(2, 2), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)

print("MultinomialNB modeli çıkart... MultinomialNB sınıfından bir nesne ürettik.")
multinomialNB = MultinomialNB()

# print("Makineyi eğitiyoruz.)
multinomialNB = multinomialNB.fit(xTrain, yTrain)

# Test veri kümemizi verdik ve  tahmin etmesini sağladık
result = multinomialNB.predict(xTest.todense())

print("MultinomialNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score = f1_score(yTest, result, average="macro")
precisionScore = precision_score(yTest, result, average="macro")
recallScore = recall_score(yTest, result, average="macro")

print('MultinomialNB F1 Score {}'.format(f1Score))
print('MultinomialNB Precision Score {}'.format(precisionScore))
print('MultinomialNB Recall Score {}'.format(recallScore))
print('MultinomialNB Başarı Oranı {}'.format(accuracy))
print("-------------------------------------------------------------")      
######################################################################################


print("MultinomialNB eğitime başla... 3-gram")
countVector = CountVectorizer(ngram_range=(3, 3), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)

print("MultinomialNB modeli çıkart... MultinomialNB sınıfından bir nesne ürettik.")
multinomialNB = MultinomialNB()

# print("Makineyi eğitiyoruz.)
multinomialNB = multinomialNB.fit(xTrain, yTrain)

# Test veri kümemizi verdik ve  tahmin etmesini sağladık
result = multinomialNB.predict(xTest.todense())

print("MultinomialNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score = f1_score(yTest, result, average="macro")
precisionScore = precision_score(yTest, result, average="macro")
recallScore = recall_score(yTest, result, average="macro")

print('MultinomialNB F1 Score {}'.format(f1Score))
print('MultinomialNB Precision Score {}'.format(precisionScore))
print('MultinomialNB Recall Score {}'.format(recallScore))
print('MultinomialNB Başarı Oranı {}'.format(accuracy))
print('#'*60)
print("-------------------------------------------------------------")      
######################################################################################

# GaussianNB : Tahmin edeceğiniz veri veya kolon sürekli (real,ondalıklı vs.) ise
print("GaussianNB eğitime başla... 1-gram")
countVector = CountVectorizer(ngram_range=(1, 1), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)
      
print("GaussianNB modeli çıkart... GaussianNB sınıfından bir nesne ürettik.")
gaussianNB = GaussianNB()

# Makineyi eğitiyoruz
gaussianNB = gaussianNB.fit(xTrain.todense(), yTrain.ravel())

# Test veri kümemizi verdik ve tahmin etmesini sağladık
result = gaussianNB.predict(xTest.todense())

print("GaussianNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score = f1_score(yTest, result, average="macro")
precisionScore = precision_score(yTest, result, average="macro")
recallScore = recall_score(yTest, result, average="macro")

print('GaussianNB F1 Score {}'.format(f1Score))
print('GaussianNB Precision Score {}'.format(precisionScore))
print('GaussianNB Recall Score {}'.format(recallScore))
print('GaussianNB Başarı Oranı {}'.format(accuracy))
print("-------------------------------------------------------------")      
######################################################################################


print("GaussianNB eğitime başla... 2-gram")
countVector = CountVectorizer(ngram_range=(2, 2), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)
      
print("GaussianNB modeli çıkart... GaussianNB sınıfından bir nesne ürettik.")
gaussianNB = GaussianNB()

# Makineyi eğitiyoruz
gaussianNB = gaussianNB.fit(xTrain.todense(), yTrain.ravel())

# Test veri kümemizi verdik ve tahmin etmesini sağladık
result = gaussianNB.predict(xTest.todense())

print("GaussianNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score = f1_score(yTest, result, average="macro")
precisionScore = precision_score(yTest, result, average="macro")
recallScore = recall_score(yTest, result, average="macro")

print('GaussianNB F1 Score {}'.format(f1Score))
print('GaussianNB Precision Score {}'.format(precisionScore))
print('GaussianNB Recall Score {}'.format(recallScore))
print('GaussianNB Başarı Oranı {}'.format(accuracy))
print("-------------------------------------------------------------")      
######################################################################################


print("GaussianNB eğitime başla... 3-gram")
countVector = CountVectorizer(ngram_range=(3, 3), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)
      
print("GaussianNB modeli çıkart... GaussianNB sınıfından bir nesne ürettik.")
gaussianNB = GaussianNB()

# Makineyi eğitiyoruz
gaussianNB = gaussianNB.fit(xTrain.todense(), yTrain.ravel())

# Test veri kümemizi verdik ve tahmin etmesini sağladık
result = gaussianNB.predict(xTest.todense())

print("GaussianNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score = f1_score(yTest, result, average="macro")
precisionScore = precision_score(yTest, result, average="macro")
recallScore = recall_score(yTest, result, average="macro")

print('GaussianNB F1 Score {}'.format(f1Score))
print('GaussianNB Precision Score {}'.format(precisionScore))
print('GaussianNB Recall Score {}'.format(recallScore))
print('GaussianNB Başarı Oranı {}'.format(accuracy))
print('#'*60)
print("-------------------------------------------------------------")      
######################################################################################


print("BernoulliNB eğitime başla... 1-gram")
countVector = CountVectorizer(ngram_range=(1, 1), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)
      
print("BernoulliNB modeli çıkart... BernoulliNB sınıfından bir nesne ürettik.")
bernoulliNB = BernoulliNB()

# Makineyi eğitiyoruz
bernoulliNB = bernoulliNB.fit(xTrain.todense(), yTrain.ravel())

# Test veri kümemizi verdik ve tahmin etmesini sağladık
result = bernoulliNB.predict(xTest.todense())
print("BernoulliNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score=f1_score(yTest, result, average="macro")
precisionScore=precision_score(yTest, result, average="macro")
recallScore=recall_score(yTest, result, average="macro")

print('BernoulliNB F1 Score {}'.format(f1Score))
print('BernoulliNB Precision Score {}'.format(precisionScore))
print('BernoulliNB Recall Score {}'.format(recallScore))
print('BernoulliNB Başarı Oranı {}'.format(accuracy))
print("-------------------------------------------------------------")   
######################################################################################


# BernoulliNB : Tahmin edeceğiniz veri veya kolon ikili ise ( Evet/Hayır , Sigara içiyor/ İçmiyor vs.)
print("BernoulliNB eğitime başla... 2-gram")
countVector = CountVectorizer(ngram_range=(2, 2), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)
      
print("BernoulliNB modeli çıkart... BernoulliNB sınıfından bir nesne ürettik.")
bernoulliNB = BernoulliNB()

# Makineyi eğitiyoruz
bernoulliNB = bernoulliNB.fit(xTrain.todense(), yTrain.ravel())

# Test veri kümemizi verdik ve tahmin etmesini sağladık
result = bernoulliNB.predict(xTest.todense())
print("BernoulliNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score=f1_score(yTest, result, average="macro")
precisionScore=precision_score(yTest, result, average="macro")
recallScore=recall_score(yTest, result, average="macro")

print('BernoulliNB F1 Score {}'.format(f1Score))
print('BernoulliNB Precision Score {}'.format(precisionScore))
print('BernoulliNB Recall Score {}'.format(recallScore))
print('BernoulliNB Başarı Oranı {}'.format(accuracy))
print("-------------------------------------------------------------")   
######################################################################################


print("BernoulliNB eğitime başla... 3-gram")
countVector = CountVectorizer(ngram_range=(3, 3), min_df=0.01, stop_words='english')
X = countVector.fit_transform(text)
xTrain, xTest, yTrain, yTest = train_test_split(X,labels,train_size=0.8,test_size=0.2,random_state=0)
      
print("BernoulliNB modeli çıkart... BernoulliNB sınıfından bir nesne ürettik.")
bernoulliNB = BernoulliNB()

# Makineyi eğitiyoruz
bernoulliNB = bernoulliNB.fit(xTrain.todense(), yTrain.ravel())

# Test veri kümemizi verdik ve tahmin etmesini sağladık
result = bernoulliNB.predict(xTest.todense())
print("BernoulliNB Confusion Matrix")
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# yTest :  Test Verisi
# result : xTest verisinden tahmin ettiğimiz  veriler
confusionMatrix = confusion_matrix(yTest, result)
print(confusionMatrix)

print("Başarı Oranı")
accuracy = accuracy_score(yTest, result)
f1Score=f1_score(yTest, result, average="macro")
precisionScore=precision_score(yTest, result, average="macro")
recallScore=recall_score(yTest, result, average="macro")

print('BernoulliNB F1 Score {}'.format(f1Score))
print('BernoulliNB Precision Score {}'.format(precisionScore))
print('BernoulliNB Recall Score {}'.format(recallScore))
print('BernoulliNB Başarı Oranı {}'.format(accuracy))
print('#'*60)
print("-------------------------------------------------------------")   
######################################################################################

print("Sequential modeli çıkart...")
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250

# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['Icerik'].values)
X = tokenizer.texts_to_sequences(data['Icerik'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(data['Kategori']).values
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.05, random_state=42)
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 5
batch_size = 128
history = model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, 
                    validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
acc = model.evaluate(xTest, yTest)
print(acc[1])

accr = model.evaluate(xTest,yTest)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

print("...Teşekkürler...\n\n")
