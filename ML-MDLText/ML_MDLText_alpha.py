# -*- coding: utf-8 -*- 
import numpy as np
import scipy as sp
import sklearn as skl
from sklearn import naive_bayes
from sklearn.base import BaseEstimator
from sklearn import linear_model
import math


import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #arquivo com as funções de seleção de features

import ML_MDLTextFunctions as ML_MDLTextFunctions



class ML_MDLText(BaseEstimator):
    """
    MDLText
    
    Attributes:
        omega: (vocabulary size) (default 2^10) 
        feature_relevance_function: function to calculate the relevance of tokens (default CF)
    """

    def __init__(self, clfClasses=None, relevanceMethod = 'CF', omega=2**10):
        # parametro usado no calculo de K
        self.alpha = 0.001 

        self.relevanceMethod = relevanceMethod
        self.omega = omega
        self.frequency = None
        self.binary_frequency = None
        self.freqTokensClass = None
        self.nTrain = None
        self.classes = []
        self.nFeatures = 0
        self.copyable_attrs = ['clfClasses', 'relevanceMethod','omega']

        self.clfClasses = clfClasses

        self.matOcurrencesClasses = None
        self.nTrainTotal = 0
        self.classes_multiclasse = None

        self.centroids = None
        self.norm_centroids = None


    def __repr__(self):
        return "ML_MDLText(clfClasses = \'%s\', relevanceMethod = \'%s\', omega = %1.2f)" %(self.clfClasses, self.relevanceMethod, self.omega)
        

    # Fit
    def fit(self, x_train, y_train, classes = None, partial_fit=False, classes_multiclasse=None):
        """
        Train the classifier
        """

        # Converte a base de dados para esparsa se não estiver em formato esparso        
        if not sp.sparse.issparse(x_train):
            x_train = sp.sparse.csr_matrix(x_train)

        # Verifica a quantidade de exemplos, atributos e classes
        nSamples = x_train.shape[0]
        nFeatures = x_train.shape[1]
        classes = np.unique( range(y_train.shape[1]) )

        
        # Treina Meta-modelo
        if self.frequency is None:
            if classes_multiclasse is None:
                classes_multiclasse = np.unique(y_train.sum(axis=1))
        
            self.clfClasses.partial_fit(x_train, y_train.sum(axis=1), classes=classes_multiclasse)
        else:
            self.clfClasses.partial_fit(x_train, y_train.sum(axis=1), classes=classes_multiclasse)

        # Inicia Variáveis 
        self.frequency = ML_MDLTextFunctions._expandMatrix(self.frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.binary_frequency = ML_MDLTextFunctions._expandMatrix(self.binary_frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.freqTokensClass = ML_MDLTextFunctions._expandMatrix(self.freqTokensClass, len(classes)-len(self.classes), 0)
        self.nTrain = ML_MDLTextFunctions._expandMatrix(self.nTrain, len(classes)-len(self.classes), 0)
        self.matOcurrencesClasses = ML_MDLTextFunctions._expandMatrix(self.matOcurrencesClasses, len(classes)-len(self.classes), len(classes)-len(self.classes))


        # Atualiza dados das classes
        for i in range( len(classes) ):

            idClass = y_train[:,i]==1 #i é o número da classe

            # Conta o número de exemplos de treinamento para cada classe (|D^|)
            self.nTrain[i] += np.count_nonzero(idClass) 

            # Soma as ocorrencias de cada token na classe atual (n)
            self.frequency[:,i] += np.asarray(x_train[idClass,:].sum(axis=0))[0] 

            # Soma as ocorrencias de todos os tokens na classe atual (n^)
            self.freqTokensClass[i] = self.frequency[:,i].sum() 
            
            aux = x_train[idClass,:]
            aux[aux!=0]=1 #convert os dados para representação binária
            
            # Frequência dos tokens (phi)
            self.binary_frequency[:,i] += np.asarray( aux.sum(axis=0) )[0]

            # Frequência dos tokens ( |D^|(c_i, c_j) )
            self.matOcurrencesClasses[i] += (y_train[idClass,:]==1).sum(axis=0) #spearman(y_train[:,i], y_train[:,j])[0]

        # Calcula a relevância dos termos
        self.featureRelevance = ML_MDLTextFunctions._calcFeatureRelevance(self.binary_frequency, nTrain=self.nTrain, relevanceMethod=self.relevanceMethod)
        
        # Calcula centroides e norma centroid
        self.centroids, self.norm_centroids = ML_MDLTextFunctions._calcCentroids(self.nTrain, self.frequency)

        # Acumula a quantidade de exemplos de treinamento, define as classes e os atributos
        self.nTrainTotal += nSamples
        self.classes = classes
        self.nFeatures = nFeatures




    # Treinamernto incremental
    def partial_fit(self, x_train, y_train, classes = None, classes_multiclasse=None):
        """
        Train the classifier
        """
        self.fit(x_train, y_train, classes, partial_fit=True, classes_multiclasse=classes_multiclasse)
       

    # Predict
    def predict(self, x_test):
        """
        Predict
        """
        
        # Converte a base de dados para esparsa se não estiver em formato esparso 
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        # Verifica a quantidade de exemplos, atributos e classes
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]

        y_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        # Buscando a quantidade de classes Método MetaC - Tang
        numClass = self.clfClasses.predict(x_test) # classificador que determina a quantidade de rótulos de cada exemplo

        for i in range( nTest ):

            # Calcula o tamanho de descrição
            lengthTextDependences = self._descriptionLengthDependence(x_test[i,:], (numClass[i] > 1))
            
            # Define as classes relevantes
            ranking = sorted(list(enumerate(lengthTextDependences)), key=lambda lt: lt[1])
            for j in range(0,numClass[i]):
                y_pred[i,ranking[j][0]] = 1
                   
            
        return y_pred


    # Predict Proba
    def predict_proba(self, x_test):
        """
        Predict
        """

        # Converte a base de dados para esparsa se não estiver em formato esparso 
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        # Verifica a quantidade de exemplos, atributos e classes
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]

        y_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        lengthTexto_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        for i in range( nTest ):
            
            # Calcula o tamanho de descrição
            lengthTextDependences = self._descriptionLengthDependence(x_test[i,:], True)
            lengthTextDependences = np.nan_to_num(lengthTextDependences)
            lengthTexto_pred[i] = list((-1)*lengthTextDependences)
            
        
        return np.array(lengthTexto_pred)



    # Tamanho de descrição
    def _descriptionLengthDependence(self, x_test, addDependence = False):

        lengthTexto = ML_MDLTextFunctions._lengthDescription(x_test, self.nTrain, self.frequency, self.binary_frequency, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha=self.alpha, featureRelevance=self.featureRelevance, relevanceMethod=self.relevanceMethod)
        
        # Define um ranking de classes
        lengthTextDependences = lengthTexto
        if addDependence:
            ranking = sorted(list(enumerate(lengthTextDependences)), key=lambda lt: lt[1])
            # Inclui peso da dependência entre as classes
            lengthTextDependences = lengthTexto + ranking[0][1] * ((1-self.matOcurrencesClasses[ranking[0][0]]/self.nTrain[ranking[0][0]]))
        
        return lengthTextDependences




if __name__ == "__main__":

    clfClasses = skl.linear_model.SGDClassifier(random_state=5)

    x_train = np.array([[0,0,0.5,0.8],[0.2,0.4,0,0],[0.3,0,0,0],[0.1,0.1,0.2,0.2]])
    x_train = sp.sparse.csr_matrix(x_train)
    
    x_test = x_train
    
    y_train = np.array([[1,0],[0,1],[1,1],[1,1]])
    y_test = y_train
    
    mdl = ML_MDLText(clfClasses=clfClasses)
    mdl.fit(x_train,y_train)
    
    y_pred = mdl.predict(x_test)
    
    