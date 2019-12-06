# -*- coding: utf-8 -*- 

import numpy as np
import scipy as sp
import sklearn as skl
import math

import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #arquivo com as funções de seleção de features


# Centroids
def _calcCentroids(nTrain, frequency, idFeature=None):

    centroids = np.nan_to_num( np.where(nTrain == 0, 0, frequency / nTrain) )
    
    if idFeature is None:
        norm_centroids = np.linalg.norm(centroids, axis=0)
    else:
        norm_centroids = np.linalg.norm(centroids[idFeature], axis=0)

    return centroids, norm_centroids



# Aumenta Tamanho da Matriz para o treinamento Online
def _expandMatrix(data, positionsBottom, positionsRight, newData=None):

    if data is None:
        if  positionsRight > 0:
            data = np.zeros( (positionsBottom, positionsRight) )
        else:
            data = np.zeros( (positionsBottom) )
    else:
        if newData is None:
            if data.ndim > 1:
                if positionsBottom > 0:
                    newData = np.zeros((positionsBottom, data.shape[1]))   #0
                else:
                    newData = np.zeros((data.shape[0],positionsRight ))    #1
            else:
                newData = np.zeros( positionsBottom )


        if data.ndim > 1 and positionsBottom > 0:
            data = ( np.vstack((data,newData)) )
        else:
            data = ( np.hstack((data,newData)) )
        
    return data


# Calcula o Tamanho de descrição
def _lengthDescription(x_test, nTrain, frequency, binary_frequency, freqTokensClass, omega, centroids, norm_centroids, alpha = 0.001, featureRelevance=None, relevanceMethod=None):

    # Returns the non-negative values of a sparse matrix
    idFeature = sp.sparse.find( x_test )[1]
    
    if len(idFeature):
        freqTokensClass[freqTokensClass == 0] = float('inf')

        # Relevância dos termos
        if featureRelevance is None:
            featureRelevance = _calcFeatureRelevance(binary_frequency[idFeature], nTrain=nTrain, relevanceMethod=relevanceMethod)
        else:
            featureRelevance = featureRelevance[idFeature]
        
        # Calcula K
        k_relevancy = 1/(1+alpha-(featureRelevance))   #np.log2

        # Calcula beta
        probToken =  ( ( (frequency[idFeature]+(1/omega))/(freqTokensClass+1) )  ).T 

        # Calcula tamanho de descrição dos termos
        tokenLength = np.ceil( -np.log2(probToken) )

        # Similaridade Cosseno
        norm_doc = sp.sparse.linalg.norm( x_test[:,idFeature] )
        
        # Similaridade cosseno
        cosine_similarity = np.where(( norm_doc * norm_centroids) == 0, 0,  ( x_test[:,idFeature] * centroids[idFeature] ) / ( norm_doc * norm_centroids ))[0]

        # Tamanho de descrição do documento
        text_length = np.sum( tokenLength * ( k_relevancy ), axis=1 ) * ( -np.log2(0.5*(cosine_similarity)) ) #1/(     np.power( cosine_similarity ,1) )#( -np.log2(0.5*(cosine_similarity + alpha)) ) #

        
    else:
        tokenLength = 0
        k_relevancy = 0
        cosine_similarity = 0

        text_length = np.where(sum(nTrain) == 0, 0, nTrain/sum(nTrain))
        text_length = -np.log2( text_length ) 


    return text_length


# Calcula a relevância dos termos
def _calcFeatureRelevance(binary_frequency, nTrain=None, relevanceMethod=None):

    if 'CF' in relevanceMethod:
        featureRelevance = fs.fatorConfidencia(binary_frequency)
    elif 'DFS' in relevanceMethod:
        featureRelevance = fs.calcula_dfs(binary_frequency, nTrain)
    else:
        featureRelevance = np.zeros(binary_frequency.shape[0])

    return featureRelevance

    