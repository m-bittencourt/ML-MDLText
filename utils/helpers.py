# -*- coding: utf-8 -*- 
import numpy as np
import sklearn as skl
import scipy
import os
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy import stats

from bs4 import BeautifulSoup

import arff

from collections import defaultdict

from sklearn import preprocessing


def import_dataset_multilabel(nomeDataset, tipoDataset, nLabels, nMinSamplesPerClass=0):
    """
    Function used to import the multilabel textual database
    
    Parameters:
    -----------
    pathDataset: string
        Is the address of database. Each line in the database, must have the format <class, message>.        
    """    
    # 'pathDataset' is the database address
    pathDataset = 'datasets/' + nomeDataset

    if ('txt' in tipoDataset):

        datasetFileText  = open(pathDataset + '_texto.txt','r', encoding='utf-8', errors='ignore') #abre arquivo para leitura
        datasetFileClass = open(pathDataset + '_classes.txt','r', encoding='utf-8', errors='ignore') #abre arquivo para leitura


    
        dataset = [] # empty list that will store the messages
        target = []
        for line, lineClass in zip(datasetFileText, datasetFileClass):
            dataset.append(line.replace('\n', ''))
            target.append(lineClass.replace('\n', '').split(','))

        datasetFileText.close()
        datasetFileClass.close()

        [classes, target] = import_multilabelTargets(target)
    
        dataset, target, classes = clean_dataset(nomeDataset, dataset, target, classes, nMinSamplesPerClass)

        target = np.array(target) #convert a list to a numpy array
        dataset = np.array(dataset) #convert a list to a numpy array

    elif ('mulan' in tipoDataset):
        if ('emotions' in nomeDataset or
            'yeast' in nomeDataset or
            'CAL500' in nomeDataset):
            load_sparse = False
        else:
            load_sparse = True
        dataset, target, classes = load_dataset_Mulan(pathDataset, nomeDataset, labelcount=nLabels, input_feature_type='float', load_sparse=load_sparse)
        dataset, target, classes = clean_dataset(nomeDataset, dataset, target, classes, nMinSamplesPerClass)

    # function used to describe the database
    #information_dataset(dataset,target,classes)

    return dataset, classes, target

def unique_rowsArray(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def import_multilabelTargets(auxTarget):
    classes = []
    for i in range( len(auxTarget) ):
        for j in range( len(auxTarget[i]) ):
            if auxTarget[i][j] not in classes:
                classes.append(auxTarget[i][j])
                
    classes.sort()
    
    target = np.zeros( (len(auxTarget),len(classes)), dtype=int )
    for i in range( len(auxTarget) ):
        for j in range( len(classes) ):
            if classes[j] in auxTarget[i]:
                target[i,j] = 1
                
    return classes, target

def clean_dataset(dataname, data,target,classes, nMinSamplesPerClass):


    if 'toxit' in dataname:
        idx =  np.where( target[:,2]==0 )[0]
        target = target[idx,:] 
        data2 = []
        for i in idx:
            data2.append(data[i])
        data = data2
    
    nData = target.sum(axis=0) 
    idx = np.where(nData>nMinSamplesPerClass)[0]
    target = target[:,idx]     
    
    classes = [classes[c] for c in range( len(classes) ) if c in idx ]
    
    # remove data that remains with no label
    idx = np.where(target.sum(axis=1)>0)[0]
    target = target[idx,:]

    if sparse.issparse(data): 
        dataset = data[idx,:]
    else:
        dataset = []
        for i in idx:
            dataset.append(data[i].replace('youidiot', 'you idiot').replace('meoff', 'me off').replace('llkill', 'll kill'))
            
    
    return dataset, target, classes

def information_dataset(data,target,classes):
        
    nLabelPerData = target.sum(axis=1)
    cardinality = np.average(nLabelPerData) #mean of the number of labels of the instances 
    
    density = cardinality/len(classes)  #mean of the number of labels of the instances that belong to the dataset divided by the number of dataset’s labels

    
    nData = target.sum(axis=0)
    for i in range( len(classes ) ):
        print('%s - %d' %(classes[i],nData[i]))
        
    print('\n\nnSamples: %d -- nFeatures: %d -- cardinality: %1.3f -- density: %1.3f' %(data.shape[0], data.shape[1], cardinality,density))

def load_dataset_Mulan(pathDataset, datasetName, labelcount, input_feature_type='float', encode_nominal=True, load_sparse=False):

    matrix = None
    endian="little"

    subDiretorios = os.listdir(pathDataset+'/')

    for filename in subDiretorios:

        if '.xml' in filename:
            fileTexto = open(pathDataset+'/'+'/'+filename, encoding='UTF-8', errors='replace' ) #encoding='cp850' ->Latin-1
            texto = fileTexto.read()
            soup = BeautifulSoup(texto, 'lxml')
                
            topicos = []
            
            # extract the text that is inside the tags <codes class="bip:topics:1.0">
            for tagCodes in soup.find_all('labels'):
                for auxTopicos in tagCodes.find_all('label'):
                    topicos.append( auxTopicos.get('name', []) )# extract the value that is inside the code tag

        elif '.arff' in filename:          

            matrixFold = None

            if not load_sparse:
                arff_frame = arff.load(open(pathDataset+'/'+'/'+filename, 'r'), encode_nominal=encode_nominal, return_type=arff.DENSE)
                matrixFold = sparse.csr_matrix(arff_frame['data'], dtype=input_feature_type)
            else:
                arff_frame = arff.load(open(pathDataset+'/'+'/'+filename, 'r'), encode_nominal=encode_nominal, return_type=arff.COO)
                data = arff_frame['data'][0]
                row = arff_frame['data'][1]
                col = arff_frame['data'][2]
                matrixFold = sparse.csr_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1))
                

            if matrix == None:
                matrix = matrixFold
            else:
                matrix = sparse.vstack((matrix,matrixFold), format='csr')
        else:
            # unknown file
            None


    X, y = None, None

    if endian == "big":
        X, y = matrix.tocsr()[:, labelcount:].tolil(), matrix.tocsr()[:, :labelcount].astype(int).tolil()
    elif endian == "little":
        X, y = matrix.tocsr()[:, :-labelcount], matrix.tocsr()[:, -labelcount:].astype(int)
    else:
        # unknown endian
        None
            
    return X, y.toarray(), topicos


#============================================
# Function to convert Text in Term-Frequency array
#============================================
def text2tf(text, add_vocabulary = False, add_Docs = False, vocabulary = None, language = None, feature_frequency = None, nDocs = None):
    """
    Do the conversion of text to an term-frequency array (TF)
    Its possible to update the vocabulary and the frequency of attributes in docuements in a incremental way, allowing that new documents be analyzed in different moments. 
    """   
    if nDocs is None:
        nDocs = 0

    # When its a new training sample, its necessary to add new terms to the vocabulary
    if add_vocabulary:

        # Initialize the vocabulary, the attributes frequency and the number of documents
        if vocabulary is None:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        if feature_frequency is None:
            feature_frequency = np.zeros((1,1)).astype(int)

        # Splits the terms of documents 
        tokenized_text = [word_tokenize(row, language=language) for row in text] # tokenized docs
        tokens = set([item for sublist in tokenized_text for item in sublist])

        #print(nDocs)
        #a = text[0].split(' ')
        #for token in tokens:
        #    for i, b in enumerate(a):
        #        if (b == token):
        #            a[i] = ''
        #for b in a:
        #    if b != '':
        #        print(b)
        
        # Add to vocabulary the terms of analyzed documents
        for word in tokens:
            try:
                feature_idx = vocabulary[word]
            except KeyError:
                continue

    # Generates the term-frequency array to texts
    vectorizer = skl.feature_extraction.text.CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, lowercase = True, binary=False, dtype=np.int32, vocabulary = vocabulary )

    tf = vectorizer.transform(text) 

    if add_Docs:

        # Updates the frequency of terms while the vocabulary changes
        # calculates the quantity of new terms/attributes to add the frequency
        if feature_frequency is None:
            new_terms = len(vocabulary)
            
        else:
            new_terms = len(vocabulary) - feature_frequency.shape[1]
        new_features = np.zeros( (1,new_terms) ).astype(int)

        # calculates/updates the frequency of attributes
        if feature_frequency is None:
            feature_frequency = (tf != 0).sum(axis=0)
            
        else:
            feature_frequency = np.hstack((feature_frequency,new_features)) + (tf != 0).sum(axis=0)
        nDocs += len(text)

        return tf, vocabulary, feature_frequency, nDocs 
    else:
        return tf

    
#============================================
#Function to convert tf to tf-idf
#============================================
def tf2tfidf(tf, df=None,  nDocs=None, normalize_tf=False, normalize_tfidf=True, return_df=False):
    """
    Do the conversion to tf_idf. 
    When used on the test phase, must be informed the frequency of the documents in training database 
    that contain each token and the quantity of training docuements. 
    
    One of the differences of this function to the sklearn.feature_extraction.text.TfidfVectorizer function
    is that it uses log in base 10 instead of a natural logarithm. Besides that, the Tf is normalized as 
    np.log10( 1+tf.data ), while ino scikit it is normalized as  1 + np.log( tf.data ). Still,
    the IDF is calculated as np.log10( (nDocs+1)/(df+1) ), while in scikit is  
    np.log(nDocs / df) + 1.0
    """    

    #se não é esparsa, converte em esparsa
    if not scipy.sparse.issparse(tf):
        tf = csc_matrix(tf)

    if normalize_tf == True:
        tf.data = np.log10( 1+tf.data )
            
    if df is None:    
        df = (tf != 0).sum(axis=0) #document frequency -- number of documents where term i occurs

    if nDocs is None:   
        nDocs = tf.shape[0] 

    #nDocs += 1 #used to avoid that idf be negative when the token appears in all documents         
    
    idf = np.log10( (nDocs+1)/(df+1) ) #-- we add 1 to avoid 0 division
    idf = csc_matrix(idf);
    
    #tf_idf = csc_matrix( (tf.shape) )

    tf_idf = tf.multiply(idf)
        
    #tf_idf = np.nan_to_num(tf_idf) #Replace nan with zero and inf with finite numbers
    #tf_idf2 = csc_matrix(tf_idf)  
        
    if normalize_tfidf==True and tf.shape[0] > 0:
        tf_idf = skl.preprocessing.normalize(tf_idf, norm='l2')
        
    if return_df == True:
        return tf_idf, df
    else:
        return tf_idf

#============