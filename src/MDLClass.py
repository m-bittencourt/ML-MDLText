# -*- coding: utf-8 -*- 
import numpy as np
import scipy as sp

import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #file with feature selection functions
from sklearn.base import BaseEstimator
from abc import abstractmethod

class MDLClass(BaseEstimator):

    def __init__(self, clfClasses=None, relevanceMethod = 'CF', omega=2**10):

        # parameter used in K calculation
        self.alpha = 0.001 

        self.relevanceMethod = relevanceMethod
        self.omega = omega
        self.frequency = None
        self.binary_frequency = None
        self.freqTokensClass = None
        self.nTrain = None
        self.classes = []
        self.nFeatures = 0
        

        self.clfClasses = clfClasses
        self.nTrainTotal = 0

        self.centroids = None
        self.norm_centroids = None


    @abstractmethod
    def fit(self, x_train, y_train, classes = None, classes_multiclasse=None):
        pass

    @abstractmethod
    def partial_fit(self, x_train, y_train, classes=None, classes_multiclasse=None):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass


    # Centroids
    def _calcCentroids(self, nTrain, frequency, idFeature=None):

        centroids = np.nan_to_num( np.where(nTrain == 0, 0, frequency / nTrain) )
        
        if idFeature is None:
            norm_centroids = np.linalg.norm(centroids, axis=0)
        else:
            norm_centroids = np.linalg.norm(centroids[idFeature], axis=0)

        return centroids, norm_centroids



    # Increase the matrix size to Online training
    def _expandMatrix(self, data, positionsBottom, positionsRight, newData=None):

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


    # Calculates the Description size
    def _lengthDescription(self, x_test, nTrain, frequency, binary_frequency, freqTokensClass, omega, centroids, norm_centroids, alpha = 0.001, featureRelevance=None, relevanceMethod=None):

        # Returns the non-negative values of a sparse matrix
        idFeature = sp.sparse.find( x_test )[1]
        
        if len(idFeature):
            freqTokensClass[freqTokensClass == 0] = float('inf')

            # Relevance of terms
            if featureRelevance is None:
                featureRelevance = self._calcFeatureRelevance(binary_frequency[idFeature], nTrain=nTrain, relevanceMethod=relevanceMethod)
            else:
                featureRelevance = featureRelevance[idFeature]
            
            # Calculates K
            k_relevancy = 1/(1+alpha-(featureRelevance))   #np.log2

            # Calculates beta
            probToken =  ( ( (frequency[idFeature]+(1/omega))/(freqTokensClass+1) )  ).T 

            # Calculates the terms description length
            tokenLength = np.ceil( -np.log2(probToken) )

            # Cosine similarity
            norm_doc = sp.sparse.linalg.norm( x_test[:,idFeature] )
            
            # Cosine similarity
            cosine_similarity = np.where(( norm_doc * norm_centroids) == 0, 0,  ( x_test[:,idFeature] * centroids[idFeature] ) / ( norm_doc * norm_centroids ))[0]

            # Document description length
            text_length = np.sum( tokenLength * ( k_relevancy ), axis=1 ) * ( -np.log2(0.5*(cosine_similarity)) ) #1/(     np.power( cosine_similarity ,1) )#( -np.log2(0.5*(cosine_similarity + alpha)) ) #

            
        else:
            tokenLength = 0
            k_relevancy = 0
            cosine_similarity = 0

            text_length = np.where(sum(nTrain) == 0, 0, nTrain/sum(nTrain))
            text_length = -np.log2( text_length ) 


        return text_length


    # Calculates the relevance of terms
    def _calcFeatureRelevance(self, binary_frequency, nTrain=None, relevanceMethod=None):

        if 'CF' in relevanceMethod:
            featureRelevance = fs.fatorConfidencia(binary_frequency)
        elif 'DFS' in relevanceMethod:
            featureRelevance = fs.calcula_dfs(binary_frequency, nTrain)
        else:
            featureRelevance = np.zeros(binary_frequency.shape[0])

        return featureRelevance



        