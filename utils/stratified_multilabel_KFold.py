import numpy as np

class stratified_multilabel_KFold():
    """
    This script is based on work by: Sechidis, Konstantinos and Tsoumakas, Grigorios and Vlahavas, Ioannis: On the stratification of 
                                     multi-label data. In: Proceedings of the 2011 European conference on Machine learning and knowledge 
                                     discovery in databases (ECML PKDD'11), volume part III, pp. 145-158 (2011)
    """

    def __init__(self, n_splits=5, shuffle=True, randomSeed=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.randomSeed = randomSeed
        
    def split(self,target):
        k = self.n_splits
        shuffle = self.shuffle
        randomSeed = self.randomSeed
        
        nLabels = target.shape[1]
        
        nDataPerLabel = target.sum(axis=0)
                                          
        desiredSplit = np.zeros( (k,nLabels) )
        desiredSplitPerFold = np.zeros(k)
        for i in range( k ):
            desiredSplitPerFold[i] = target.shape[0]/k
            desiredSplit[i,:] = nDataPerLabel/k
                        
        #print(nDataPerLabel)    
        #print(desiredSplit) 
        
        nD = target.shape[0] #number of remaining examples
    
        if shuffle is True:
            if randomSeed is None:
                idxRemainingExamples = np.random.RandomState().permutation(range(nD))
            else:
                idxRemainingExamples = np.random.RandomState(randomSeed).permutation(range(nD))
                
            idxRemainingExamples = list(idxRemainingExamples)
        else:
            idxRemainingExamples = list(range( nD ))
        
        subsets = []
        for i in range(k):
            subsets.append([])
                         
        while nD > 0:
            #Find the label with the fewest (but at least one) remaining examples
            idxSort = np.argsort(nDataPerLabel)
            for i in range(nLabels):
                if nDataPerLabel[ idxSort[i] ] > 0:
                    idx_minLabel = idxSort[i]
                    break
            
            auxIdxLabel = np.where(target[idxRemainingExamples,idx_minLabel]==1)[0] #the indexes of the label idx_minLabel
            idxLabel = [ idxRemainingExamples[i] for i in auxIdxLabel ]
            
            for idx in idxLabel:
                idxMaxSubset = np.argmax( desiredSplit[:,idx_minLabel] ) #Find the subset(s) with the largest number of desired examples for this
                #Check if there was a tie.
                #-----------
                auxIdx = np.where( desiredSplit[:,idx_minLabel]==desiredSplit[idxMaxSubset,idx_minLabel] )[0] #id dos lables empatados
                if len(auxIdx)>1:
                    auxMax = np.argmax( desiredSplitPerFold[ auxIdx ] ) #Find the subset(s) with the largest number of desired examples for this
                    idxMaxSubset = auxIdx[ auxMax ]    
            
                subsets[idxMaxSubset].append( idx )
                
                desiredSplitPerFold[idxMaxSubset] -= 1
                desiredSplit[idxMaxSubset,:] -= target[idx,:]
                
                nDataPerLabel -= target[idx,:]
                
                auxIdx = idxRemainingExamples.index(idx)
                idxRemainingExamples = idxRemainingExamples[0:auxIdx]+idxRemainingExamples[auxIdx+1:]
                
                nD = nD-1
                
                
        #for i in range(k):
        #    print('\n', target[subsets[i]])
            
        #Joining the folds in training and testing
        #separa os índices de treinamento e de teste
        folds_final = np.zeros( k,dtype='object')
        train_index = np.zeros( k,dtype='object')
        test_index = np.zeros( k,dtype='object')
        for nf in range(k):
            #Joining the other folds to create the training data
    
            #the current fold will be used for testing. The other folds will be used for training
            test_index[nf] = subsets[nf]
    
            train_index[nf] = []
            for nf2 in range(k):
                if nf2 != nf:
                    train_index[nf] = train_index[nf]+subsets[nf2]
                
            folds_final[nf] = np.array( [train_index[nf],test_index[nf]] )
                        
        return folds_final

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

