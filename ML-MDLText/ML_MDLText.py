# -*- coding: utf-8 -*- 

import numpy as np
import scipy as sp
import sklearn as skl
from sklearn import naive_bayes
from sklearn import linear_model
import math
from matplotlib import pyplot as mp

import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #arquivo com as funções de seleção de features

from skmultilearn.base import MLClassifierBase
from scipy.sparse import lil_matrix

import sklearn as skl
import myFunctions

import ML_MDLTextFunctions

class ML_MDLText(MLClassifierBase):
    """
    MDLText
    
    Attributes:
        omega: (vocabulary size) (default 2^10) 
        feature_relevance_function: function to calculate the relevance of tokens (default CF)
    """

    def __init__(self, clfClasses, relevanceMethod='CFLP', omega=2**10, omegaComb=2**10, calc_feature_relevance=True, sampling=100, n_samples_pruned=0, consider_samples_pruned=False):
        # parametro usado no calculo de K
        self.alpha = 0.001 
        self.gamma1 = 0.95
        self.gamma2 = 0.20

        self.relevanceMethod = relevanceMethod
        self.omega = omega
        self.omegaComb = omegaComb
        self.frequency = None
        self.binary_frequency = None
        self.freqTokensClass = None
        self.nTrain = None
        self.classes = []
        self.nFeatures = 0
        self.copyable_attrs = ['clfClasses', 'relevanceMethod', 'omega', 'omegaComb', 'calc_feature_relevance', 'sampling', 'n_samples_pruned', 'consider_samples_pruned']

        self.clfClasses = clfClasses

        self.nTrainTotal = 0
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

        self.centroids = None
        self.norm_centroids = None

        self.centroidsComb = None
        self.norm_centroidsComb = None

    def __repr__(self):
        return "MDLText(omega = %1.2f, omegaComb = %1.2f, relevanceMethod = \'%s\', clfClasses = \'%s\', sampling = %i, , n_samples_pruned = %i)" %(self.omega, self.omegaComb, self.relevanceMethod, self.clfClasses, self.sampling, self.n_samples_pruned)


    # Fit
    def fit(self, x_train, y_train, classes=None, classes_multiclasse=None):
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



        # Inicia Variáveis 
        # Variáveis das classes
        self.count = ML_MDLTextFunctions._expandMatrix(self.count, len(classes)-len(self.classes), 0)

        self.frequency = ML_MDLTextFunctions._expandMatrix(self.frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.binary_frequency = ML_MDLTextFunctions._expandMatrix(self.binary_frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.freqTokensClass = ML_MDLTextFunctions._expandMatrix(self.freqTokensClass, len(classes)-len(self.classes), 0)
        self.nTrain = ML_MDLTextFunctions._expandMatrix(self.nTrain, len(classes)-len(self.classes), 0)

        # Variáveis das combinações de Classes
        all_lc, n_samples_all_lc = np.unique(y_train, return_counts=True, axis=0)
        lc = all_lc
        n_samples_comb = n_samples_all_lc
        #idx_comb = (n_samples_all_lc > self.n_samples_pruned)
        #lc, n_samples_comb = all_lc[idx_comb], n_samples_all_lc[idx_comb]
        
        # Inicia Variáveis Combinação
        firstExecution = False
        if self.lc.shape[0] == 0:
            self.lc = lc
            self.frequencyLP = ML_MDLTextFunctions._expandMatrix(self.frequencyLP, nFeatures-self.nFeatures, self.lc.shape[0])
            self.binary_frequencyLP = ML_MDLTextFunctions._expandMatrix(self.binary_frequencyLP, nFeatures-self.nFeatures, self.lc.shape[0])
            self.freqTokensClassLP = ML_MDLTextFunctions._expandMatrix(self.freqTokensClassLP, self.lc.shape[0], 0)
            self.nTrainLP = ML_MDLTextFunctions._expandMatrix(self.nTrainLP, self.lc.shape[0], 0)

            firstExecution = True




        # Extrai Informações relevantes de cada classe
        all_idClasse = (y_train == 1).T

        #conta o número de exemplos de treinamento para cada classe
        self.nTrain += np.count_nonzero(all_idClasse, axis=1) 

        for i, idClasse in enumerate(all_idClasse):
            # Soma as ocorrencias de cada token na classe atual (n)
            self.frequency[:, i] += np.asarray(x_train[idClasse, :].sum(axis=0))[0]

            # Frequência dos tokens (phi)
            self.binary_frequency[:, i] += np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]    # Conta o numero de atributos que aparecem nos documentos da classe

        # Soma as ocorrencias de todos os tokens na classe atual (n^)
        self.freqTokensClass = self.frequency.sum(axis=0) #soma as ocorrencias de todos os tokens na classe atual
        



        # Extrai Informações relevantes de cada combinação
        for i in range(0, lc.shape[0]):
            vector = lc[i]

            # Incluindo novas combinações
            if firstExecution:
                indexComb = i
            else:
                self.frequencyLP = ML_MDLTextFunctions._expandMatrix(self.frequencyLP, nFeatures-self.nFeatures, 0)
                self.binary_frequencyLP = ML_MDLTextFunctions._expandMatrix(self.binary_frequencyLP, nFeatures-self.nFeatures, 0)
                self.nFeatures = nFeatures

                try:
                    indexComb = self.lc.tolist().index(vector.tolist())
                except ValueError:
                    self.lc = np.append(self.lc, [vector.tolist()], axis=0)
                    self.frequencyLP = ML_MDLTextFunctions._expandMatrix(self.frequencyLP, 0, 1)
                    self.binary_frequencyLP = ML_MDLTextFunctions._expandMatrix(self.binary_frequencyLP, 0, 1)
                    self.freqTokensClassLP = ML_MDLTextFunctions._expandMatrix(self.freqTokensClassLP, 1, 0)
                    self.nTrainLP = ML_MDLTextFunctions._expandMatrix(self.nTrainLP, 1, 0)

                    indexComb = self.lc.shape[0] - 1

            idClasse = (y_train == vector).all(axis=1) #i é o número da combinação

            #conta o número de exemplos de treinamento para cada combinação
            self.nTrainLP[indexComb] += n_samples_comb[i]

            # Soma as ocorrencias de cada token na combinação atual (n)
            self.frequencyLP[:, indexComb] += np.asarray(x_train[idClasse, :].sum(axis=0))[0]
            # Frequência dos tokens de cada combinação(phi)
            self.binary_frequencyLP[:, indexComb] += np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]  

            # Soma as ocorrencias de todos os tokens na combinação atual (n^)
            self.freqTokensClassLP[indexComb] = self.frequencyLP[:,indexComb].sum(axis=0) #soma as ocorrencias de todos os tokens na classe atual


        #if self.consider_samples_pruned:
            # Transfere os exemplos das combinações cortadas para as demais combinações
        #    idx_pruned = (n_samples_all_lc <= self.n_samples_pruned)
        #    lc_pruned, n_samples_pruned = all_lc[idx_pruned], n_samples_all_lc[idx_pruned]
        #    for i, vector in enumerate(lc_pruned):
        #        idClasse = (y_train == vector).all(axis=1)
        #        idx = np.where(vector == 1)[0]
        #        j = (lc[:, idx] == 1).any(axis=1) # todas combinções com quantidade maior de exemplos para herdar os exemplos da que será cortada

        #        self.nTrainLP[j] += n_samples_pruned[i]
        #        self.frequencyLP[:, j] = (self.frequencyLP[:, j].T + np.asarray(x_train[idClasse, :].sum(axis=0))[0]).T
        #        self.binary_frequencyLP[:, j] = (self.binary_frequencyLP[:, j].T + np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]).T  # Conta o numero de atributos que aparecem nos documentos da classe

        

        # Diferenciação do Calculo da relevancia por por classe ou por combinação de classes
        if 'LP' in self.relevanceMethod:
            self.binary_freq = self.binary_frequencyLP
        else:
            self.binary_freq = self.binary_frequency


        # Calcula relevancia dos termos
        if self.calc_feature_relevance:
            self.featureRelevance = ML_MDLTextFunctions._calcFeatureRelevance(self.binary_freq, nTrain=self.nTrain, relevanceMethod=self.relevanceMethod)

        # Calcula centroids
        self.centroids, self.norm_centroids = ML_MDLTextFunctions._calcCentroids(self.nTrain, self.frequency)
        self.centroidsComb, self.norm_centroidsComb = ML_MDLTextFunctions._calcCentroids(self.nTrainLP, self.frequencyLP, idFeature = None)  


        # Treina Meta-Modelo
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

        
        # Acumula a quantidade de exemplos de treinamento, define as classes e os atributos
        self.nTrainTotal += nSamples
        self.classes = classes
        self.nFeatures = nFeatures



        # Geranção das curvas gaussianas para os dados
        self.classesInterval = np.array(range(0, self.classesMeta.max()+1))
        # Atualizando a estrutura de sigma e 
        self.sd = ML_MDLTextFunctions._expandMatrix(self.sd, len(self.classesInterval), 0)

        # Prediz as classes com o Meta-modelo
        y_pred = (np.round(self.clfClasses.predict(x_train_meta)).astype(int) )  #np.ones( (nSamples), dtype=int )#

        # Obtem o calculo da matrix de confusão para F-Medida
        cm = skl.metrics.confusion_matrix(y_meta, y_pred, self.classesInterval)
        self.tp += cm.diagonal()
        self.fn += cm.sum(axis=1)-(cm.diagonal())
        self.fp += cm.sum(axis=0)-(cm.diagonal())
        self.tn += cm.sum() - (cm.diagonal()) - (cm.sum(axis=0)-(cm.diagonal())) - (cm.sum(axis=1)-(cm.diagonal()))

        precision = (self.tp / (self.tp+self.fp))
        recall = (self.tp / (self.tp+self.fn))
        f_medida = (2 * ((precision * recall)/(precision + recall)))
        self.sd = -np.log2((    self.gamma1 * np.nan_to_num( f_medida )    )**2)
        

        # Confiança das primeiras posições do ranking considerando os exemplos de treinamento
        # Caso esse procedimento seja feito com uma amostragem:
        n_sampling = np.ceil(x_train.shape[0]*(self.sampling/100)).astype(int)
        np.random.seed(1)
        index = np.random.randint(0, x_train.shape[0], size=n_sampling)

        # Definindo o valor de cm
        if n_sampling > 0:
            x_sampling = x_train[index, :]
            y_sampling = y_train[index, :]
            nTest = x_sampling.shape[0]

            text_length_pred = np.zeros((nTest, len(classes)), dtype=int)
            
            for i in range(0, nTest):
                text_length = ML_MDLTextFunctions._lengthDescription(x_sampling[i, :], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, featureRelevance=self.featureRelevance, relevanceMethod=self.relevanceMethod)
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

        # Converte a base de dados para esparsa se não estiver em formato esparso 
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]

        y_pred = lil_matrix(  np.zeros( (nTest, len(self.classes)), dtype=int )  )

        # Buscando a quantidade de classes Método MetaC - Tang
        numClass = (np.round( self.clfClasses.predict(x_test) )).astype(int) #np.ones( (nTest), dtype=int )## classificador que determina a quantidade de rótulos de cada exemplo

        # Calculando as Centroides das classes
        for i in range( nTest ):
            
            # Calculo tamanho de Descrição
            text_length = ML_MDLTextFunctions._lengthDescription(x_test[i,:], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha = self.alpha, featureRelevance = self.featureRelevance, relevanceMethod=self.relevanceMethod)
            
            # Define um ranking de classes
            ranking_dep = sorted(range(len(text_length)), key=text_length.__getitem__)

            # Correlação Baseada nas combinações
            if len(text_length) > 0:

                classes = []

                classInterval = np.array( range(((-1)* numClass[i] ), ((len(self.classesInterval)- numClass[i] )+1) ) )
                
                # Define o valor de G da classe definida em numClass[i] para cada um dos tamanhos de labelset possíveis
                gaussian = self._gaussianFunction(classInterval, 0, self.sd[  numClass[i]  ])
                # Obtem os labelset size que serão considerados
                labelsetSizes = classInterval[ np.round(gaussian,5)>0.0 ] +  numClass[i] 


                # Seleciona as combinações de acordo com o tamanho da combinação e as classes mais provaveis do ranking
                try:
                    index =  ( (self.lc.sum(axis=1)  >=  labelsetSizes.min()) & (self.lc.sum(axis=1) <=  labelsetSizes.max()) ) & (  (self.lc[:,ranking_dep[:len(self.cm)] ]==1).any(axis=1) )  #(self.lc.sum(axis=1)==m) &  ( ((self.lc.sum(axis=1) >=  labelsetSizes.min()) & (self.lc.sum(axis=1) <=  labelsetSizes.max()) ).all(axis=0) )         
                except:
                    index = []

                classes = list( self.lc[index])

                if (len(classes) > 1):
                    # Seleciona as estatísticas e calcula o tamanho da descrição para cada uma delas
                    nTrain, frequency, freqTokensClass, binary_frequency = self.nTrainLP[index], self.frequencyLP[:,index], self.freqTokensClassLP[index], self.binary_frequencyLP[:,index]#self._calcFrequencias(self.frequencyLP, self.binary_frequencyLP, self.nTrainLP, self.lc, classes)
                    lengthTextoDep = ML_MDLTextFunctions._lengthDescription(x_test[i,:], nTrain, frequency, self.binary_freq, freqTokensClass, self.omegaComb, self.centroidsComb[:,index], self.norm_centroidsComb[index], self.alpha, featureRelevance=self.featureRelevance, relevanceMethod=self.relevanceMethod)


                    # Aplica penalidade baseada no retorno da função gaussiana
                    penalidadeGauss = 0
                    if len(classes) > 0:
                        labelsetSizes = np.sum( classes, axis=1) - numClass[i]
                        gaussian = self._gaussianFunction( labelsetSizes, 0, self.sd[ numClass[i] ])
                        penalidadeGauss = self.gamma2*(1-gaussian)

                    lengthTextoDep = lengthTextoDep * (1+penalidadeGauss)

                    # Seleciona o labelset com menor tamanho de descrição
                    ranking_dep = sorted(range(len(lengthTextoDep)), key=lengthTextoDep.__getitem__)
                    y_pred[i] = classes[ranking_dep[0]]

                # Quando há apenas um labelset possível
                elif (len(classes) == 1):
                    y_pred[i, :] = self.lc[index][0]

                # Quando não há nenhum labelset possível, são selecionadas as classes com menor tamanho de descrição
                else:
                    y_pred[i, ranking_dep[:numClass[i]]] = 1
            else:
                # Quando não há nenhum labelset possível, são selecionadas as classes com menor tamanho de descrição
                y_pred[i, ranking_dep[:numClass[i]]] = 1

        return y_pred


    def predict_proba(self, x_test):
        """
        Predict
        """

        # Converte a base de dados para esparsa se não estiver em formato esparso   
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csr_matrix(x_test)

        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]
    
        textLength_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        for i in range( nTest ):
            
            # Calcula o tamanho de descrição e inverte os valores
            textLength = ML_MDLTextFunctions._lengthDescription(x_test[i,:], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha = alpha, featureRelevance = self.featureRelevance, relevanceMethod=self.relevanceMethod)
            textLength_pred[i] = list((-1)*textLength)
        
        return np.array(textLength_pred)


    # Calcula a Função Gaussiana
    def _gaussianFunction(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



if __name__ == "__main__":

    mdl = ML_MDLText(clfClasses=skl.linear_model.SGDClassifier(random_state=5))

    x_train = np.array([[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	                    [0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
	                    [0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0],
	                    [0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
	                    [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0],
	                    [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1]])

    x_train = sp.sparse.csr_matrix(   sp.sparse.csr_matrix(x_train)   )
    x_test = np.array([[0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0]])

    x_test = sp.sparse.csr_matrix(   sp.sparse.csr_matrix(x_test)   )

    #x_train, df_train = myFunctions.tf2tfidf(x_train, normalize_tf=True, normalize_tfidf=True, return_df=True)
    #x_test = myFunctions.tf2tfidf(x_test, df=df_train, nDocs=x_train.shape[0], normalize_tf=True, normalize_tfidf=True)
    
    
    y_train =  np.array( [[1,1],[0,1],[1,0],[1,0],[1,1],[0,1]] )
    y_test = [1]
    
    for i in range(0,x_train.shape[0]):
        if i%2==1:
            mdl.partial_fit( x_train[(i-1):i+1], y_train[(i-1):i+1], [1,2] )
    
    #y_pred = mdl.predict_proba(x_test)
    y_pred = mdl.predict(x_test)
    
    #print(y_pred)
    