# -*- coding: utf-8 -*- 
import numpy as np
import scipy as sp
import sklearn as skl

from MDLClass import MDLClass

import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #file with feature selection functions



class ML_MDLText_alpha(MDLClass):
    """
    MDLText
    
    Attributes:
        omega: (vocabulary size) (default 2^10) 
        feature_relevance_function: function to calculate the relevance of tokens (default CF)
    """

    def __init__(self, clfClasses=None, relevanceMethod = 'CF', omega=2**10):
        
        super().__init__(clfClasses, relevanceMethod, omega)

        self.copyable_attrs = ['clfClasses', 'relevanceMethod','omega']
        self.matOcurrencesClasses = None
        self.classes_multiclasse = None


    def __repr__(self):
        return "ML_MDLText(clfClasses = \'%s\', relevanceMethod = \'%s\', omega = %1.2f)" %(self.clfClasses, self.relevanceMethod, self.omega)
        

    # Fit
    def fit(self, x_train, y_train, classes = None, classes_multiclasse=None):
        """
        Train the classifier
        """

        # Convert the database to sparse if its not in sparse format
        if not sp.sparse.issparse(x_train):
            x_train = sp.sparse.csr_matrix(x_train)

        # Check the samples, attributes and classes quantity
        nSamples = x_train.shape[0]
        nFeatures = x_train.shape[1]
        classes = np.unique( range(y_train.shape[1]) )

        # Train the Meta-model
        if self.frequency is None:
            if classes_multiclasse is None:
                classes_multiclasse = np.unique(y_train.sum(axis=1))
        
            self.clfClasses.partial_fit(x_train, y_train.sum(axis=1), classes=classes_multiclasse)
        else:
            self.clfClasses.partial_fit(x_train, y_train.sum(axis=1), classes=classes_multiclasse)

        # Initialize variables
        self.frequency = super()._expandMatrix(self.frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.binary_frequency = super()._expandMatrix(self.binary_frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.freqTokensClass = super()._expandMatrix(self.freqTokensClass, len(classes)-len(self.classes), 0)
        self.nTrain = super()._expandMatrix(self.nTrain, len(classes)-len(self.classes), 0)
        self.matOcurrencesClasses = super()._expandMatrix(self.matOcurrencesClasses, len(classes)-len(self.classes), len(classes)-len(self.classes))


        # Updates the classes data
        for i in range( len(classes) ):

            idClass = y_train[:,i]==1 #i é o número da classe

            # Count the number of training samples for each class (|D^|)
            self.nTrain[i] += np.count_nonzero(idClass) 

            # Sum the occurrences of each token in the current class (n)
            self.frequency[:,i] += np.asarray(x_train[idClass,:].sum(axis=0))[0] 

            # Sum the occurrences of all tokens in the current class (n^)
            self.freqTokensClass[i] = self.frequency[:,i].sum() 
            
            aux = x_train[idClass,:]
            aux[aux!=0]=1 #convert data to binary representation
            
            # Tokens frequencies (phi)
            self.binary_frequency[:,i] += np.asarray( aux.sum(axis=0) )[0]

            # Tokens frequencies ( |D^|(c_i, c_j) )
            self.matOcurrencesClasses[i] += (y_train[idClass,:]==1).sum(axis=0) #spearman(y_train[:,i], y_train[:,j])[0]

        # Calculates the relevance of terms
        self.featureRelevance = super()._calcFeatureRelevance(self.binary_frequency, nTrain=self.nTrain, relevanceMethod=self.relevanceMethod)
        
        # Calculates the centroids and centroid norm
        self.centroids, self.norm_centroids = super()._calcCentroids(self.nTrain, self.frequency)

        # Acumulates the quantity of training samples, defines the classes and attributes
        self.nTrainTotal += nSamples
        self.classes = classes
        self.nFeatures = nFeatures




    # Incremental training
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
        
        # Convert the database to sparse if its not in sparse format
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        # Check the samples, attributes and classes quantity
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]

        y_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        # Searching the quantity of classes MetaC - Tang method
        numClass = self.clfClasses.predict(x_test) # classificador que determina a quantidade de rótulos de cada exemplo

        for i in range( nTest ):

            # Calculates the description length
            lengthTextDependences = self._descriptionLengthDependence(x_test[i,:], (numClass[i] > 1))
            
            # Defines the relevant classes
            ranking = sorted(list(enumerate(lengthTextDependences)), key=lambda lt: lt[1])
            for j in range(0,numClass[i]):
                y_pred[i,ranking[j][0]] = 1
                   
            
        return y_pred


    # Predict Proba
    def predict_proba(self, x_test):
        """
        Predict
        """

        # Convert the database to sparse if its not in sparse format
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        # Check the samples, attributes and classes quantity
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]

        y_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        lengthTexto_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        for i in range( nTest ):
            
            # Calculates the description length
            lengthTextDependences = self._descriptionLengthDependence(x_test[i,:], True)
            lengthTextDependences = np.nan_to_num(lengthTextDependences)
            lengthTexto_pred[i] = list((-1)*lengthTextDependences)
            
        
        return np.array(lengthTexto_pred)



    # Description length
    def _descriptionLengthDependence(self, x_test, addDependence = False):

        lengthTexto = super()._lengthDescription(x_test, self.nTrain, self.frequency, self.binary_frequency, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha=self.alpha, featureRelevance=self.featureRelevance, relevanceMethod=self.relevanceMethod)
        
        # Defines a ranking of classes
        lengthTextDependences = lengthTexto
        if addDependence:
            ranking = sorted(list(enumerate(lengthTextDependences)), key=lambda lt: lt[1])
            # Includes a weigth to dependency between classes
            lengthTextDependences = lengthTexto + ranking[0][1] * ((1-self.matOcurrencesClasses[ranking[0][0]]/self.nTrain[ranking[0][0]]))
        
        return lengthTextDependences
