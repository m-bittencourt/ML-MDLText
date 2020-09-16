# -*- coding: utf-8 -*- 

import numpy as np
import scipy as sp
import sklearn as skl
from sklearn import linear_model

from MDLClass import MDLClass

import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #file with feature selection functions

from scipy.sparse import lil_matrix


class ML_MDLText(MDLClass):
    """
    MDLText
    
    Attributes:
        omega: (vocabulary size) (default 2^10) 
        feature_relevance_function: function to calculate the relevance of tokens (default CF)
    """

    def __init__(self, clfClasses, relevanceMethod='CFLP', omega=2**10, omegaComb=2**10, calc_feature_relevance=True, sampling=100, n_samples_pruned=0, consider_samples_pruned=False):

        super().__init__(clfClasses, relevanceMethod, omega)

        # parameter used in K calculation
        self.gamma1 = 0.95
        self.gamma2 = 0.20
        self.omegaComb = omegaComb

        self.copyable_attrs = ['clfClasses', 'relevanceMethod', 'omega', 'omegaComb', 'calc_feature_relevance', 'sampling', 'n_samples_pruned', 'consider_samples_pruned']

        self.featureRelevance = None
        self.calc_feature_relevance = calc_feature_relevance

        self.frequencyLP = None
        self.nTrainLP = None
        self.lc = np.array([])
        self.binary_frequencyLP = None
        self.freqTokensClassLP = None

        self.binary_freq = None

        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

        self.classesMeta = None
        self.classesInterval = None
        self.sd = None

        self.one_error = None

        self.cm = None
        self.count = None

        self.sampling = sampling
        self.n_samples_pruned = n_samples_pruned
        self.consider_samples_pruned = consider_samples_pruned

        self.centroidsComb = None
        self.norm_centroidsComb = None

    def __repr__(self):
        return "MDLText(omega = %1.2f, omegaComb = %1.2f, relevanceMethod = \'%s\', clfClasses = \'%s\', sampling = %i, , n_samples_pruned = %i)" %(self.omega, self.omegaComb, self.relevanceMethod, self.clfClasses, self.sampling, self.n_samples_pruned)


    # Fit
    def fit(self, x_train, y_train, classes=None, classes_multiclasse=None):
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



        # Initialize variables 
        # Classes variables
        self.count = super()._expandMatrix(self.count, len(classes)-len(self.classes), 0)

        self.frequency = super()._expandMatrix(self.frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.binary_frequency = super()._expandMatrix(self.binary_frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.freqTokensClass = super()._expandMatrix(self.freqTokensClass, len(classes)-len(self.classes), 0)
        self.nTrain = super()._expandMatrix(self.nTrain, len(classes)-len(self.classes), 0)

        # Labelset variables
        all_lc, n_samples_all_lc = np.unique(y_train, return_counts=True, axis=0)
        lc = all_lc
        n_samples_comb = n_samples_all_lc
        #idx_comb = (n_samples_all_lc > self.n_samples_pruned)
        #lc, n_samples_comb = all_lc[idx_comb], n_samples_all_lc[idx_comb]
        
        # Initialize labelset variables
        firstExecution = False
        if self.lc.shape[0] == 0:
            self.lc = lc
            self.frequencyLP = super()._expandMatrix(self.frequencyLP, nFeatures-self.nFeatures, self.lc.shape[0])
            self.binary_frequencyLP = super()._expandMatrix(self.binary_frequencyLP, nFeatures-self.nFeatures, self.lc.shape[0])
            self.freqTokensClassLP = super()._expandMatrix(self.freqTokensClassLP, self.lc.shape[0], 0)
            self.nTrainLP = super()._expandMatrix(self.nTrainLP, self.lc.shape[0], 0)

            firstExecution = True




        # Retrieve relevant informations of each label
        all_idClasse = (y_train == 1).T

        # Get the number of training samples for each class
        self.nTrain += np.count_nonzero(all_idClasse, axis=1) 

        for i, idClasse in enumerate(all_idClasse):
            # Sums each token occurrences in the current class (n)
            self.frequency[:, i] += np.asarray(x_train[idClasse, :].sum(axis=0))[0]

            # Frequencies of tokens (phi)
            self.binary_frequency[:, i] += np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]    # Get the number of attributes in class documents

        # Sum all tokens occurrences in the current class (n^)
        self.freqTokensClass = self.frequency.sum(axis=0) # Sum all tokens occurrences in the current class
        



        # Get relevant information of each labelset
        for i in range(0, lc.shape[0]):
            vector = lc[i]

            # Including new labelsets
            if firstExecution:
                indexComb = i
            else:
                self.frequencyLP = super()._expandMatrix(self.frequencyLP, nFeatures-self.nFeatures, 0)
                self.binary_frequencyLP = super()._expandMatrix(self.binary_frequencyLP, nFeatures-self.nFeatures, 0)
                self.nFeatures = nFeatures

                try:
                    indexComb = self.lc.tolist().index(vector.tolist())
                except ValueError:
                    self.lc = np.append(self.lc, [vector.tolist()], axis=0)
                    self.frequencyLP = super()._expandMatrix(self.frequencyLP, 0, 1)
                    self.binary_frequencyLP = super()._expandMatrix(self.binary_frequencyLP, 0, 1)
                    self.freqTokensClassLP = super()._expandMatrix(self.freqTokensClassLP, 1, 0)
                    self.nTrainLP = super()._expandMatrix(self.nTrainLP, 1, 0)

                    indexComb = self.lc.shape[0] - 1

            idClasse = (y_train == vector).all(axis=1) #i is the labelset number

            # Count the sample number for each labelset
            self.nTrainLP[indexComb] += n_samples_comb[i]

            # Sum the occurrences of each token in the current labelset (n)
            self.frequencyLP[:, indexComb] += np.asarray(x_train[idClasse, :].sum(axis=0))[0]
            # Token frequencies for each labelset (phi)
            self.binary_frequencyLP[:, indexComb] += np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]  

            # Sum all token occurrences in the current labelset (n^)
            self.freqTokensClassLP[indexComb] = self.frequencyLP[:,indexComb].sum(axis=0) # Sum all token occurrences in the current label


        #if self.consider_samples_pruned:
            # Transfer the pruned labelsets samples to other labelsets
        #    idx_pruned = (n_samples_all_lc <= self.n_samples_pruned)
        #    lc_pruned, n_samples_pruned = all_lc[idx_pruned], n_samples_all_lc[idx_pruned]
        #    for i, vector in enumerate(lc_pruned):
        #        idClasse = (y_train == vector).all(axis=1)
        #        idx = np.where(vector == 1)[0]
        #        j = (lc[:, idx] == 1).any(axis=1) # All labelsets with greater samples quantity to inherit the samples from that will be cut

        #        self.nTrainLP[j] += n_samples_pruned[i]
        #        self.frequencyLP[:, j] = (self.frequencyLP[:, j].T + np.asarray(x_train[idClasse, :].sum(axis=0))[0]).T
        #        self.binary_frequencyLP[:, j] = (self.binary_frequencyLP[:, j].T + np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]).T  # Get the number of attributes in class documents

        

        # Calculus of relevance considering class or labelset
        if 'LP' in self.relevanceMethod:
            self.binary_freq = self.binary_frequencyLP
        else:
            self.binary_freq = self.binary_frequency


        # Calculates the relevance of terms
        if self.calc_feature_relevance:
            self.featureRelevance = super()._calcFeatureRelevance(self.binary_freq, nTrain=self.nTrain, relevanceMethod=self.relevanceMethod)

        # Calculate centroids
        self.centroids, self.norm_centroids = super()._calcCentroids(self.nTrain, self.frequency)
        self.centroidsComb, self.norm_centroidsComb = super()._calcCentroids(self.nTrainLP, self.frequencyLP, idFeature = None)  


        # Training Meta-Models
        y_meta = y_train.sum(axis=1)

        allClasses, nData = np.unique( y_meta, return_counts=True)   
        if classes_multiclasse is None:
            self.classesMeta = allClasses[ (nData>=(1)) ]
        else:
            self.classesMeta = classes_multiclasse#allClasses[ (nData>=(1)) ]

        idx = np.where([(y in self.classesMeta) for y in y_meta])[0]

        y_meta = y_meta[idx]
        x_train_meta = x_train[idx]
        self.clfClasses.partial_fit(x_train_meta, y_meta, self.classesMeta)  

        
        # Accumulates the amount of training samples, define the labels and attributes
        self.nTrainTotal += nSamples
        self.classes = classes
        self.nFeatures = nFeatures



        # Generation of gaussian curves to data
        self.classesInterval = np.array(range(0, self.classesMeta.max()+1))
        # Updating sigma e structure 
        self.sd = super()._expandMatrix(self.sd, len(self.classesInterval), 0)

        # Predict classes with Meta-model
        y_pred = (np.round(self.clfClasses.predict(x_train_meta)).astype(int) )  #np.ones( (nSamples), dtype=int )#

        # Obtain the confusion matrix calculus to F-Measure
        cm = skl.metrics.confusion_matrix(y_meta, y_pred, self.classesInterval)
        self.tp += cm.diagonal()
        self.fn += cm.sum(axis=1)-(cm.diagonal())
        self.fp += cm.sum(axis=0)-(cm.diagonal())
        self.tn += cm.sum() - (cm.diagonal()) - (cm.sum(axis=0)-(cm.diagonal())) - (cm.sum(axis=1)-(cm.diagonal()))

        precision = np.nan_to_num((self.tp / (self.tp+self.fp)) )
        recall = np.nan_to_num((self.tp / (self.tp+self.fn)) )
        f_medida = np.nan_to_num( (2 * ((precision * recall)/(precision + recall))) )
        self.sd = -np.log2((    self.gamma1 * np.nan_to_num( f_medida )    )**2)
        

        # Confidence of first positions of ranking considering the training samples
        # If this procedure is done with a sample:
        n_sampling = np.ceil(x_train.shape[0]*(self.sampling/100)).astype(int)
        np.random.seed(1)
        index = np.random.randint(0, x_train.shape[0], size=n_sampling)

        # Defining the value of cm
        if n_sampling > 0:
            x_sampling = x_train[index, :]
            y_sampling = y_train[index, :]
            nTest = x_sampling.shape[0]

            text_length_pred = np.zeros((nTest, len(classes)), dtype=int)
            
            for i in range(0, nTest):
                text_length = super()._lengthDescription(x_sampling[i, :], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, featureRelevance=self.featureRelevance, relevanceMethod=self.relevanceMethod)
                text_length_pred[i] = np.argsort(text_length)

                for c in np.array((range(0, len(classes)))):
                    if y_sampling[ i, text_length_pred[i, :(c+1) ]  ].sum() > 0:
                        self.count[c:] = self.count[c:] + 1
                        break

            self.cm = np.array( range(0, np.where( (self.count/self.nTrainTotal > 0.9) )[0][0] +1) )
        else:
            self.cm = np.array([0])

        return self


    # partial_fit
    def partial_fit(self, x_train, y_train, classes=None, classes_multiclasse=None):
        """
        Train the classifier
        """
        self.fit(x_train, y_train, classes, classes_multiclasse=self.classesMeta)
       


    # predict
    def predict(self, x_test):
        
        """
        Predict
        """

        # Convert the database to sparse if its not in sparse format
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]

        y_pred = lil_matrix(  np.zeros( (nTest, len(self.classes)), dtype=int )  )

        # Searching the quantity of classes MetaC - Tang method
        numClass = (np.round( self.clfClasses.predict(x_test) )).astype(int) #np.ones( (nTest), dtype=int )## classifier that defines the quantity of labels of each sample

        # Calculating the Centroids of classes
        for i in range( nTest ):
            
            # Description length calculus
            text_length = super()._lengthDescription(x_test[i,:], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha = self.alpha, featureRelevance = self.featureRelevance, relevanceMethod=self.relevanceMethod)
            
            # Defines a ranking of classes
            ranking_dep = sorted(range(len(text_length)), key=text_length.__getitem__)

            # Correlation based on labelsets
            if len(text_length) > 0:

                classes = []

                classInterval = np.array( range(((-1)* numClass[i] ), ((len(self.classesInterval)- numClass[i] )+1) ) )
                
                # Defines the value of G of the defined label in numClass[i] for each size of possible labelsets
                gaussian = self._gaussianFunction(classInterval, 0, self.sd[  numClass[i]  ])
                # Obtain the labelset size that will be considered
                labelsetSizes = classInterval[ np.round(gaussian,5)>0.0 ] +  numClass[i] 


                # Select the labelsets according to the size of labelset and the most probable classes of ranking
                try:
                    index =  ( (self.lc.sum(axis=1)  >=  labelsetSizes.min()) & (self.lc.sum(axis=1) <=  labelsetSizes.max()) ) & (  (self.lc[:,ranking_dep[:len(self.cm)] ]==1).any(axis=1) )  #(self.lc.sum(axis=1)==m) &  ( ((self.lc.sum(axis=1) >=  labelsetSizes.min()) & (self.lc.sum(axis=1) <=  labelsetSizes.max()) ).all(axis=0) )         
                except:
                    index = []

                classes = list( self.lc[index])

                if (len(classes) > 1):
                    # Select the stats and calculates the description size for each one of them
                    nTrain, frequency, freqTokensClass, binary_frequency = self.nTrainLP[index], self.frequencyLP[:,index], self.freqTokensClassLP[index], self.binary_frequencyLP[:,index]#self._calcFrequencias(self.frequencyLP, self.binary_frequencyLP, self.nTrainLP, self.lc, classes)
                    lengthTextoDep = super()._lengthDescription(x_test[i,:], nTrain, frequency, self.binary_freq, freqTokensClass, self.omegaComb, self.centroidsComb[:,index], self.norm_centroidsComb[index], self.alpha, featureRelevance=self.featureRelevance, relevanceMethod=self.relevanceMethod)


                    # Aplies a penalty based on the return of gaussian function
                    penalidadeGauss = 0
                    if len(classes) > 0:
                        labelsetSizes = np.sum( classes, axis=1) - numClass[i]
                        gaussian = self._gaussianFunction( labelsetSizes, 0, self.sd[ numClass[i] ])
                        penalidadeGauss = self.gamma2*(1-gaussian)

                    lengthTextoDep = lengthTextoDep * (1+penalidadeGauss)

                    # Select the labelset with the minimum description length
                    ranking_dep = sorted(range(len(lengthTextoDep)), key=lengthTextoDep.__getitem__)
                    y_pred[i] = classes[ranking_dep[0]]

                # When theres only one available labelset
                elif (len(classes) == 1):
                    y_pred[i, :] = self.lc[index][0]

                # When theres no labelset available, the labels with the minimum description length are selected
                else:
                    y_pred[i, ranking_dep[:numClass[i]]] = 1
            else:
                # When theres no labelset available, the labels with the minimum description length are selected
                y_pred[i, ranking_dep[:numClass[i]]] = 1

        return y_pred


    def predict_proba(self, x_test):
        """
        Predict
        """

        # Convert the database to sparse if its not in sparse format
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]
    
        textLength_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        for i in range( nTest ):
            
            # Calculates the description size and invert the values
            textLength = super()._lengthDescription(x_test[i,:], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha = self.alpha, featureRelevance = self.featureRelevance, relevanceMethod=self.relevanceMethod)
            textLength_pred[i] = list((-1)*textLength)
        
        return np.array(textLength_pred)


    # Calculates the Gaussian function
    def _gaussianFunction(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



    