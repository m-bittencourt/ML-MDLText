# -*- coding: utf-8 -*- 

import os #para listar arquivos e diretorios
import re #regular expression
import numpy as np
import sklearn as skl

import scipy
from scipy.io import savemat
from scipy import sparse, io
from scipy.sparse import csc_matrix

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

from bs4 import BeautifulSoup

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer #english
from nltk.stem import RSLPStemmer #portuguese
from nltk.stem import snowball #other languages

import unicodedata #remover acentos


#funcao para tratar o texto
def trataTexto(text, stopWords = False, stemming = False, regexType = 0, corpusLanguage = 'english'):
    """
    Realiza as etapas de pre-processamento no texto, tais como stemming e remoção de stopWords
    
    Parameters:
    -----------
    text: string
        O texto que será tratado. 

    stopWords: boolean
        True: remove as stopwords dos textos
        False: não remove as stopwords dos textos

    stemming: boolean
        True: aplica stemming no texto
        False: não aplica stemming no texto

    regexType: integer
        0: qualquer caractere que não seja alfanumérico ou underline é substituido pelo caractere espaço
        1: qualquer caractere que não seja alfabético é substituido pelo caractere espaço  

    corpusLanguage: string
        Indica qual o idioma do texto. Algumas etapas de processamento, como stopwords ou stemming podem ser prejudicadas se for considerado um idioma incorreto.      
    """ 
   
    if regexType==0:
        regex = re.compile('\W') #Matches any single letter, digit or underscore
    elif regexType==1: 
        regex = re.compile('[^A-Za-z]') #only letters, remove numbers and others special character
        
    text = re.sub(regex, " ", text) #
    
    #text = remove_accents(text) #remove os acentos
    text = text.lower() # Convert to lower case

    #se stopWords==1, remove as stopWords
    if stopWords==True: 
        words = text.split() # Split into words
        # Remove stop words from "words"
        words = [w for w in words if not w in nltk.corpus.stopwords.words(corpusLanguage)] 
        text = " ".join( words )

    #se stemming==1, use stemming
    if stemming==True:
        words = text.split() # Split into words

        if corpusLanguage == 'english':
            stemmer_method = PorterStemmer()  
        elif corpusLanguage == 'portuguese':
            stemmer_method = RSLPStemmer()    
        elif corpusLanguage == 'dutch':
            stemmer_method = snowball.DutchStemmer()
        elif corpusLanguage == 'french':
             stemmer_method = snowball.FrenchStemmer()
        elif corpusLanguage == 'german':
            stemmer_method = snowball.GermanStemmer()
        elif corpusLanguage == 'italian':
            stemmer_method = snowball.ItalianStemmer()
        elif corpusLanguage == 'spanish':
            stemmer_method = snowball.SpanishStemmer()
        elif corpusLanguage == 'spanish-latam': #spanish latin american
            stemmer_method = snowball.SpanishStemmer()
                
        words = [ stemmer_method.stem(w) for w in words ]
        text = " ".join( words )
    
    return text

#function to remove accents, umlauts, etc  
def remove_accents(text): 
    """
    Função usada para remover acentos no texto. Exemplos: "à" se torna "a"; "á" se torna "a".
    
    Parameters:
    -----------
    text: string
        O texto que será tratado. 
    """
    
    nfkd_form = unicodedata.normalize('NFKD', text)
    
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

