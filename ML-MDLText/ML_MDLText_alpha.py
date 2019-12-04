# -*- coding: utf-8 -*- 

import numpy as np
import scipy as sp
import sklearn as skl
from sklearn import naive_bayes
from sklearn.base import BaseEstimator
import math

import time

import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #arquivo com as funções de seleção de features




class ML_MDLText(BaseEstimator):
    """
    MDLText
    
    Attributes:
        omega: (vocabulary size) (default 2^10) 
        feature_relevance_function: function to calculate the relevance of tokens (default CF)
    """

    def __init__(self, clfClasses=None, relevanceMethod = 'CF', omega=2**10):
        self.relevanceMethod = relevanceMethod
        self.omega = omega
        self.frequencia = None
        self.frequencia_binaria = None
        self.freqTokensClasse = None
        self.nTrain = None
        self.classes = None
        self.nFeatures = None
        self.copyable_attrs = ['clfClasses', 'relevanceMethod','omega']

        self.clfClasses = clfClasses

        self.MatCorrelacaoClasses = None
        self.nTrainTotal = 0

    def __repr__(self):
        return "MDLText(clfClasses = \'%s\, omega = %1.2f, relevanceMethod = \'%s\')" %(self.clfClasses, self.omega, self.relevanceMethod)
        
    def fit(self, x_train, y_train, classes = None, partial_fit=False, classes_multiclasse=None):
        """
        Train the classifier
        """
        print(self.omega)
        print(self.clfClasses)

        
        if not sp.sparse.issparse(x_train):
            x_train = sp.sparse.csc_matrix(x_train)

        nSamples = x_train.shape[0]
        nFeatures = x_train.shape[1]
        classes = np.unique( range(y_train.shape[1]) )

        
        if self.frequencia is None:
            if classes_multiclasse is None:
                classes_multiclasse = np.unique(y_train.sum(axis=1))
        
            self.clfClasses.fit(x_train, y_train.sum(axis=1))#, classes=classes_multiclasse)
        else:
            self.clfClasses.fit(x_train, y_train.sum(axis=1))#), classes=classes_multiclasse)

        if self.frequencia is None: 
            self.frequencia = np.zeros( (nFeatures,len(classes)) )    
        else:
            # Opção necessária para o aprendizado Online - Atualização de novas classes e Atributos
            self.frequencia = self._expandMatrix(self.frequencia, nFeatures-self.nFeatures, 0)
            self.frequencia = self._expandMatrix(self.frequencia, len(classes)-len(self.classes), 1)

        if self.frequencia_binaria is None:
            self.frequencia_binaria = np.zeros( (nFeatures,len(classes)) )    
        else:
            # Opção necessária para o aprendizado Online - Atualização de novas classes e Atributos
            self.frequencia_binaria = self._expandMatrix(self.frequencia_binaria, nFeatures-self.nFeatures, 0)
            self.frequencia_binaria = self._expandMatrix(self.frequencia_binaria, len(classes)-len(self.classes), 1)

        if self.freqTokensClasse is None:
            self.freqTokensClasse = np.zeros( len(classes) )   
        else:
            # Opção necessária para o aprendizado Online - Atualização de novas classes e Atributos
            self.freqTokensClasse = self._expandMatrix(self.freqTokensClasse, len(classes)-len(self.classes), 1)

        if self.nTrain is None:   
            self.nTrain = np.zeros( len(classes) )
        else:
            # Opção necessária para o aprendizado Online - Atualização de novas classes e Atributo
            self.nTrain = self._expandMatrix(self.nTrain, len(classes)-len(self.classes), 1)


        if self.MatCorrelacaoClasses is None: 
            self.MatCorrelacaoClasses = np.zeros( (len(classes), len(classes)))    
        else:
            # Opção necessária para o aprendizado Online - Atualização de novas classes
            self.MatCorrelacaoClasses = self._expandMatrix(self.MatCorrelacaoClasses, len(classes)-len(self.classes), 0)
            self.MatCorrelacaoClasses = self._expandMatrix(self.MatCorrelacaoClasses, len(classes)-len(self.classes), 1)
            

        for i in range( len(classes) ):
            idClasse = y_train[:,i]==1 #i é o número da classe
            self.nTrain[i] += np.count_nonzero(idClasse) #conta o número de exemplos de treinamento para cada classe   
            #idClasse = np.where(y_train==i)#i é o número da classe
            self.frequencia[:,i] += np.asarray(x_train[idClasse,:].sum(axis=0))[0] 
            self.freqTokensClasse[i] = self.frequencia[:,i].sum() #soma as ocorrencias de todos os tokens na classe atual
            
            aux = x_train[idClasse,:]
            aux[aux!=0]=1 #convert os dados para representação binária
            
            self.frequencia_binaria[:,i] += np.asarray( aux.sum(axis=0) )[0]

            self.MatCorrelacaoClasses[i] += (y_train[idClasse,:]==1).sum(axis=0)#spearman(y_train[:,i], y_train[:,j])[0]

        self.featureRelevance = fs.fatorConfidencia(self.frequencia_binaria)
        
        
        #print('Time train: ', time.time() - startTime)
        self.nTrainTotal += nSamples
        
        self.classes = classes
        self.nFeatures = nFeatures

    def _expandMatrix(self, data, positions, axis):
        if positions > 0:
            if data.ndim > 1:
                if axis == 0:
                    a = np.zeros((positions, data.shape[1-axis]))
                    data = ( np.vstack((data,a)) )
                else:
                    a = np.zeros((data.shape[1-axis],positions ))
                    data = ( np.hstack((data,a)) )
            else:
                a = np.zeros( positions )
                data = ( np.hstack((data,a)) )
        
        return data

    def partial_fit(self, x_train, y_train, classes = None, classes_multiclasse=None):
        """
        Train the classifier
        """
        self.fit(x_train, y_train, classes, partial_fit=True, classes_multiclasse=classes_multiclasse)
       
    def predict(self, x_test):
        """
        Predict
        """
        
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csc_matrix(x_test)

        startTime = time.time() #iniciar o timer
        
        alpha = 0.001 # parametro usado no calculo to tamanho da descricao para o DFS, Gain e outros
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]
    
        #for i in range( len(self.classes) ):
        #    if self.nTrain[i] > 0:
        #        centroids[:,i] = self.frequencia[:,i] / self.nTrain[i]

        centroids = np.where(self.nTrain == 0, 0, self.frequencia / self.nTrain)
                 
        norm_centroids = np.linalg.norm(centroids, axis=0)

        y_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        # Buscando a quantidade de classes Método MetaC - Tang
        numClass = self.clfClasses.predict(x_test) # classificador que determina a quantidade de rótulos de cada exemplo

        for i in range( nTest ):
            idFeature = sp.sparse.find( x_test[i,:] )[1]#returns the non-negative values of a sparse matrix
            
            if len(idFeature) > 0:

                if self.relevanceMethod=='CF':
                    featureRelevance = self.featureRelevance[idFeature]#fs.fatorConfidencia(self.frequencia_binaria[idFeature,:])#self.featureRelevance[idFeature]
                elif self.relevanceMethod=='DFS':
                    featureRelevance = fs.calcula_dfs(self.frequencia_binaria[idFeature,:],self.nTrain)
                else:
                    featureRelevance = np.zeros(len(idFeature))
                    alpha=0
        		
                k_relevancy = 1/(1+alpha-featureRelevance)
                                                          
                lengthTexto = np.zeros( len(self.classes) )
                
                probToken = ( (self.frequencia[idFeature]+(1/self.omega))/(self.freqTokensClasse+1) ).T#(self.frequencia[idFeature,:]+(1/self.omega))/(self.freqTokensClasse+1)
                lengthToken = np.ceil( -np.log2(probToken) )

                norm_doc = sp.sparse.linalg.norm( x_test[i,idFeature] )

                cosine_similarity = np.where(( norm_doc * norm_centroids) == 0, 0,  ( x_test[i,idFeature] * centroids[idFeature] ) / ( norm_doc * norm_centroids ))[0]

                lengthTexto = np.sum( lengthToken * ( k_relevancy ), axis=1 ) * ( -np.log2(0.5*cosine_similarity) )
                #for j in range( len(self.classes) ):

                    #if self.nTrain[j]>0: #só faz a media se algum documento tiver sido usado para treinamento %%%se a classe não tem nenhum dado de treinamento ela é penalizada com
                    #    cosine_similarity[j] = ( x_test[i,idFeature] * centroids[idFeature,j] ) / ( norm_doc * norm_centroids[j] );
                        #cosine_similarity[j] = skl.metrics.pairwise.cosine_similarity(x_test[i,:], centroids[:,j])          

                    #lengthTexto[j] = np.sum( lengthToken[j] * ( k_relevancy ) ) * ( -np.log2(0.5*cosine_similarity[j]) )
                
            else:
                lengthTexto = np.where(sum(self.nTrain) == 0, 0, self.nTrain/sum(self.nTrain))
                lengthTexto = -np.log2( lengthTexto ) #probabilidade da classe

            
            ValorRanking = lengthTexto
            if numClass[i] > 1:
                ranking = sorted(list(enumerate(ValorRanking)), key=lambda lt: lt[1])
                ValorRanking = lengthTexto + ranking[0][1] * ((1-self.MatCorrelacaoClasses[ranking[0][0]]/self.nTrain[ranking[0][0]]))
            
            
            ranking = sorted(list(enumerate(ValorRanking)), key=lambda lt: lt[1])
            for j in range(0,numClass[i]):
                y_pred[i,ranking[j][0]] = 1
                   
            
        return y_pred


    def predict_proba(self, x_test):
        """
        Predict
        """
        
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csc_matrix(x_test)

        startTime = time.time() #iniciar o timer
        
        alpha = 0.001 # parametro usado no calculo to tamanho da descricao para o DFS, Gain e outros
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]
    
        #for i in range( len(self.classes) ):
        #    if self.nTrain[i] > 0:
        #        centroids[:,i] = self.frequencia[:,i] / self.nTrain[i]

        centroids = np.where(self.nTrain == 0, 0, self.frequencia / self.nTrain)
                 
        norm_centroids = np.linalg.norm(centroids, axis=0)

        y_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        lengthTexto_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        # Buscando a quantidade de classes Método MetaC - Tang
        numClass = self.clfClasses.predict(x_test) # classificador que determina a quantidade de rótulos de cada exemplo

        for i in range( nTest ):
            idFeature = sp.sparse.find( x_test[i,:] )[1]#returns the non-negative values of a sparse matrix
            
            if len(idFeature) > 0:

                if self.relevanceMethod=='CF':
                    featureRelevance = fs.fatorConfidencia(self.frequencia_binaria[idFeature,:])#self.featureRelevance[idFeature]
                elif self.relevanceMethod=='DFS':
                    featureRelevance = fs.calcula_dfs(self.frequencia_binaria[idFeature,:],self.nTrain)
                else:
                    featureRelevance = np.zeros(len(idFeature))
                    alpha=0
        		
                k_relevancy = 1/(1+alpha-featureRelevance)
                                                          
                lengthTexto = np.zeros( len(self.classes) )
                
                probToken = ( (self.frequencia[idFeature]+(1/self.omega))/(self.freqTokensClasse+1) ).T#(self.frequencia[idFeature,:]+(1/self.omega))/(self.freqTokensClasse+1)
                lengthToken = np.ceil( -np.log2(probToken) )

                norm_doc = sp.sparse.linalg.norm( x_test[i,idFeature] )

                cosine_similarity = np.where(( norm_doc * norm_centroids) == 0, 0,  ( x_test[i,idFeature] * centroids[idFeature] ) / ( norm_doc * norm_centroids ))[0]

                lengthTexto = np.sum( lengthToken * ( k_relevancy ), axis=1 ) * ( -np.log2(0.5*cosine_similarity) )
                #for j in range( len(self.classes) ):

                    #if self.nTrain[j]>0: #só faz a media se algum documento tiver sido usado para treinamento %%%se a classe não tem nenhum dado de treinamento ela é penalizada com
                    #    cosine_similarity[j] = ( x_test[i,idFeature] * centroids[idFeature,j] ) / ( norm_doc * norm_centroids[j] );
                        #cosine_similarity[j] = skl.metrics.pairwise.cosine_similarity(x_test[i,:], centroids[:,j])          

                    #lengthTexto[j] = np.sum( lengthToken[j] * ( k_relevancy ) ) * ( -np.log2(0.5*cosine_similarity[j]) )
                
            else:
                lengthTexto = np.where(sum(self.nTrain) == 0, 0, self.nTrain/sum(self.nTrain))
                lengthTexto = -np.log2( lengthTexto ) #probabilidade da classe
                    
            
            ValorRanking = lengthTexto
            if numClass[i] > 1:
                ranking = sorted(list(enumerate(ValorRanking)), key=lambda lt: lt[1])
                ValorRanking = lengthTexto + ranking[0][1] * ((1-self.MatCorrelacaoClasses[ranking[0][0]]/self.nTrain[ranking[0][0]]))
            
            
            ValorRanking = np.nan_to_num(ValorRanking)
            lengthTexto_pred[i] = list((-1)*ValorRanking)
            
            #===========================================================
        
        return np.array(lengthTexto_pred)





if __name__ == "__main__":
    x_train = np.array([[0,0,0.5,0.8],[0.2,0.4,0,0],[0.3,0,0,0],[0.1,0.1,0.2,0.2]])
    x_train = sp.sparse.csr_matrix(x_train)
    
    x_test = x_train
    
    y_train = np.array([[1,0],[0,1],[1,1],[1,1]])
    y_test = y_train
    
    mdl = MDLText()
    mdl.fit(x_train,y_train)
    
    y_pred = mdl.predict(x_test)
    
    