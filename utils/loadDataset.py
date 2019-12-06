# -*- coding: utf-8 -*- 
import numpy as np
import sklearn as skl
import re #regular expression

import os #para listar arquivos e diretorios
import scipy as sp
from scipy import sparse
import arff

from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

import time, timeit

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer #english
from nltk.stem import RSLPStemmer #portuguese
from nltk.stem import snowball #other languages

import sys
sys.path.append('utils')

def import_dataset_multilabel(nomeDataset, tipoDataset, nLabels, nMinSamplesPerClass=0):
    """
    Função usada para importar a base de dados textual multilabel
    
    Parameters:
    -----------
    pathDataset: string
        É o endereço da base de dados. Cada linha da base de dados, deve ter o formato <classe, mensagem>.        
    """    
    # 'pathDataset' é o endereço da base de dados
    pathDataset = 'datasets/' + nomeDataset

    if ('txt' in tipoDataset):

        datasetFileText  = open(pathDataset + '_texto.txt','r', encoding='utf-8', errors='ignore') #abre arquivo para leitura
        datasetFileClass = open(pathDataset + '_classes.txt','r', encoding='utf-8', errors='ignore') #abre arquivo para leitura


    
        dataset = [] #lista vazia que ira guardar as mensagens
        target = []
        for line, lineClass in zip(datasetFileText, datasetFileClass):
            dataset.append(line.replace('\n', ''))
            target.append(lineClass.replace('\n', '').split(','))

        datasetFileText.close()
        datasetFileClass.close()

        [classes, target] = import_multilabelTargets(target)
    
        dataset, target, classes = clean_dataset(nomeDataset, dataset, target, classes, nMinSamplesPerClass)

        target = np.array(target) #convert a lista para uma array do numpy
        dataset = np.array(dataset) #convert a lista para uma array do numpy

    elif ('mulan' in tipoDataset):
        if ('emotions' in nomeDataset or
            'yeast' in nomeDataset or
            'CAL500' in nomeDataset):
            load_sparse = False
        else:
            load_sparse = True
        dataset, target, classes = load_dataset_Mulan(pathDataset, nomeDataset, labelcount=nLabels, input_feature_type='float', load_sparse=load_sparse)
        dataset, target, classes = clean_dataset(nomeDataset, dataset, target, classes, nMinSamplesPerClass)

    # função usada para descrever a base de dados
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
    
    #remove os dados que ficaram sem nenhum label
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
            
            #extrai o texto que está dentro das tags <codes class="bip:topics:1.0">
            for tagCodes in soup.find_all('labels'):
                for auxTopicos in tagCodes.find_all('label'):
                    topicos.append( auxTopicos.get('name', []) )# extrai o valor que está dentro da tag code

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


if __name__ == "__main__":
    
    pathDataset = 'D:/Share/Mestrado/experiments/datasets'
    datasetName = 'enron'
    X, y, classes = load_dataset_Mulan(pathDataset, datasetName, labelcount=53, input_feature_type='float', load_sparse=False)

