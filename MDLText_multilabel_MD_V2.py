# -*- coding: utf-8 -*- 

import numpy as np
import scipy as sp
import sklearn as skl
from sklearn import naive_bayes
from sklearn import linear_model
import math
from matplotlib import pyplot as mp

import time

import sys
sys.path.append('../../python_commom/')
import featureRelevance as fs #arquivo com as funções de seleção de features

from skmultilearn.base import MLClassifierBase
from scipy.sparse import lil_matrix

import sklearn as skl
import myFunctions

class MDLText(MLClassifierBase):
    """
    MDLText
    
    Attributes:
        omega: (vocabulary size) (default 2^10) 
        feature_relevance_function: function to calculate the relevance of tokens (default CF)
    """

    def __init__(self, clfClasses, relevance_method='CFLP', omega=2**10, omegaComb=2**10, calc_feature_relevance=True, sampling=100, n_samples_pruned=0, consider_samples_pruned=False):

        self.relevance_method = relevance_method
        self.omega = omega
        self.omegaComb = omegaComb
        self.frequency = None
        self.binary_frequency = None
        self.freqTokensClass = None
        self.nTrain = None
        self.classes = []
        self.nFeatures = 0
        self.copyable_attrs = ['clfClasses', 'relevance_method', 'omega', 'omegaComb', 'calc_feature_relevance', 'sampling', 'n_samples_pruned', 'consider_samples_pruned']

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
        self.classesIntervalo = None
        self.sd = None

        self.one_error = None

        self.rank_position = None
        self.count = None

        self.sampling = sampling
        self.n_samples_pruned = n_samples_pruned
        self.consider_samples_pruned = consider_samples_pruned
        #print(self.n_samples_pruned)

        self.centroids = None
        self.norm_centroids = None

        self.centroidsComb = None
        self.norm_centroidsComb = None

    def __repr__(self):
        return "MDLText(omega = %1.2f, omegaComb = %1.2f, relevanceMethod = \'%s\', clfClasses = \'%s\', sampling = %i, , n_samples_pruned = %i)" %(self.omega, self.omegaComb, self.relevance_method, self.clfClasses, self.sampling, self.n_samples_pruned)


    # Funções externas
    def fit(self, x_train, y_train, classes=None, classes_multiclasse=None):
        """
        Train the classifier
        """

        times = {}
        times.update({'EstatisticaClasse': 0})
        times.update({'EstatisticaComb': 0})
        times.update({'TreinoMetaClassificador': 0})
        times.update({'RelevanciaTermos': 0})
        times.update({'CurvaGaussiana': 0})
        times.update({'PosicaoConfiancaRanking': 0})

        
        if not sp.sparse.issparse(x_train):
            x_train = sp.sparse.csc_matrix(x_train)


        nSamples = x_train.shape[0]
        nFeatures = x_train.shape[1]
        classes = np.unique(range(y_train.shape[1]))

        timer_start = time.time()

        # Opção necessária para o aprendizado Online - Atualização de novas classes e Atributos
        self.count = self._expandMatrix(self.count, len(classes)-len(self.classes), 0)

        self.frequency = self._expandMatrix(self.frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.binary_frequency = self._expandMatrix(self.binary_frequency, nFeatures-self.nFeatures, len(classes)-len(self.classes))
        self.freqTokensClass = self._expandMatrix(self.freqTokensClass, len(classes)-len(self.classes), 0)
        self.nTrain = self._expandMatrix(self.nTrain, len(classes)-len(self.classes), 0)

        # Extrai Informações relevantes de cada classe
        all_idClasse = (y_train == 1).T
        self.nTrain += np.count_nonzero(all_idClasse, axis=1) #conta o número de exemplos de treinamento para cada classe

        for i, idClasse in enumerate(all_idClasse):
            self.frequency[:, i] += np.asarray(x_train[idClasse, :].sum(axis=0))[0]
            self.binary_frequency[:, i] += np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]    # Conta o numero de atributos que aparecem nos documentos da classe

        self.freqTokensClass = self.frequency.sum(axis=0) #soma as ocorrencias de todos os tokens na classe atual
        times = self._compTime(times, 'EstatisticaClasse', timer_start)


        # Informações das combinações de Classes
        timer_start = time.time()
        all_lc, n_samples_all_lc = np.unique(y_train, return_counts=True, axis=0)
        lc = all_lc
        n_samples_comb = n_samples_all_lc
        #idx_comb = (n_samples_all_lc > self.n_samples_pruned)
        #lc, n_samples_comb = all_lc[idx_comb], n_samples_all_lc[idx_comb]
        
        firstExecution = False
        if self.lc.shape[0] == 0:
            self.lc = lc
            self.frequencyLP = self._expandMatrix(self.frequencyLP, nFeatures-self.nFeatures, self.lc.shape[0])
            self.binary_frequencyLP = self._expandMatrix(self.binary_frequencyLP, nFeatures-self.nFeatures, self.lc.shape[0])
            self.freqTokensClassLP = self._expandMatrix(self.freqTokensClassLP, self.lc.shape[0], 0)
            self.nTrainLP = self._expandMatrix(self.nTrainLP, self.lc.shape[0], 0)

            firstExecution = True


        # Extrai Informações relevantes de cada classe
        for i in range(0, lc.shape[0]):
            v = lc[i]

            if firstExecution:
                indexComb = i
            else:
                self.frequencyLP = self._expandMatrix(self.frequencyLP, nFeatures-self.nFeatures, 0)
                self.binary_frequencyLP = self._expandMatrix(self.binary_frequencyLP, nFeatures-self.nFeatures, 0)
                self.nFeatures = nFeatures

                try:
                    indexComb = self.lc.tolist().index(v.tolist())
                except ValueError:
                    self.lc = np.append(self.lc, [v.tolist()], axis=0)
                    self.frequencyLP = self._expandMatrix(self.frequencyLP, 0, 1)
                    self.binary_frequencyLP = self._expandMatrix(self.binary_frequencyLP, 0, 1)
                    self.freqTokensClassLP = self._expandMatrix(self.freqTokensClassLP, 1, 0)
                    self.nTrainLP = self._expandMatrix(self.nTrainLP, 1, 0)

                    indexComb = self.lc.shape[0] - 1

            idClasse = (y_train == v).all(axis=1) #i é o número da classe

            self.nTrainLP[indexComb] += n_samples_comb[i]
            self.frequencyLP[:, indexComb] += np.asarray(x_train[idClasse, :].sum(axis=0))[0]
            self.binary_frequencyLP[:, indexComb] += np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]    # Conta o numero de atributos que aparecem nos documentos da classe

            self.freqTokensClassLP[indexComb] = self.frequencyLP[:,indexComb].sum(axis=0) #soma as ocorrencias de todos os tokens na classe atual


        #if self.consider_samples_pruned:
            # Transfere os exemplos das combinações cortadas para as demais combinações
        #    idx_pruned = (n_samples_all_lc <= self.n_samples_pruned)
        #    lc_pruned, n_samples_pruned = all_lc[idx_pruned], n_samples_all_lc[idx_pruned]
        #    for i, v in enumerate(lc_pruned):
        #        idClasse = (y_train == v).all(axis=1)
        #        idx = np.where(v == 1)[0]
        #        j = (lc[:, idx] == 1).any(axis=1) # todas combinções com quantidade maior de exemplos para herdar os exemplos da que será cortada

                #self.nTrainLP[j] += n_samples_pruned[i]
                #self.frequencyLP[:, j] = (self.frequencyLP[:, j].T + np.asarray(x_train[idClasse, :].sum(axis=0))[0]).T
                #self.binary_frequencyLP[:, j] = (self.binary_frequencyLP[:, j].T + np.asarray((x_train[idClasse] != 0).sum(axis=0))[0]).T  # Conta o numero de atributos que aparecem nos documentos da classe

        
        times = self._compTime(times, 'EstatisticaComb', timer_start)



        # Diferenciação do Calculo da relevancia por por classe ou por combinação de classes
        timer_start = time.time()
        if 'LP' in self.relevance_method:
            self.binary_freq = self.binary_frequencyLP
        else:
            self.binary_freq = self.binary_frequency


        # Calcula relevancia dos termos
        if self.calc_feature_relevance:
            self.featureRelevance = self._calc_feature_relevance(self.binary_freq, nTrain=self.nTrain)
        times = self._compTime(times, 'RelevanciaTermos', timer_start)


        # Treina Meta-Classificador
        timer_start = time.time()
        y_meta = y_train.sum(axis=1)

        allClasses, nData = np.unique( y_meta, return_counts=True)   
        if classes_multiclasse is None:
            self.classesMeta = allClasses[ (nData>=(1)) ]
        else:
            self.classesMeta = classes_multiclasse#allClasses[ (nData>=(1)) ]

        idx = np.where([(y in self.classesMeta) for y in y_meta])[0]

        y_meta = y_meta[idx]
        x_train_meta = x_train[idx]

        self.classesIntervalo = np.array(range(0, self.classesMeta.max()+1))
        self.clfClasses.fit(x_train_meta, y_meta)#, self.classesMeta)  .partial_fit(x_train_meta, y_meta, self.classesMeta)#
        times = self._compTime(times, 'TreinoMetaClassificador', timer_start)
        


        # Geranção das curvas gaussianas para os dados
        timer_start = time.time()

        self.sd = self._expandMatrix(self.sd, len(self.classesIntervalo), 0)
        prob_classe = np.zeros((len(self.classesIntervalo)))

        y_pred = (np.round(self.clfClasses.predict(x_train_meta)).astype(int) )  #np.ones( (nSamples), dtype=int )#

        cm = skl.metrics.confusion_matrix(y_meta, y_pred, self.classesIntervalo)

        self.tp += cm.diagonal()
        self.fn += cm.sum(axis=1)-(cm.diagonal())
        self.fp += cm.sum(axis=0)-(cm.diagonal())
        self.tn += cm.sum() - (cm.diagonal()) - (cm.sum(axis=0)-(cm.diagonal())) - (cm.sum(axis=1)-(cm.diagonal()))

        precision = (self.tp / (self.tp+self.fp))
        recall = (self.tp / (self.tp+self.fn))
        prob_classe = 0.95 * np.nan_to_num(2 * ((precision * recall)/(precision + recall)))
        self.sd = -np.log2((prob_classe)**2)
        #print(self.sd)
        times = self._compTime(times, "CurvaGaussiana", timer_start)



        # Confiança das primeiras posições do ranking considerando os exemplos de treinamento
        timer_start = time.time()
        n_sampling = np.ceil(x_train.shape[0]*(self.sampling/100)).astype(int)
        #print(self.sampling, n_sampling)

        np.random.seed(1)
        index = np.random.randint(0, x_train.shape[0], size=n_sampling)

        self.nTrainTotal += nSamples

        self.centroids, self.norm_centroids = self._calcCentroids(self.nTrain, self.frequency)
        
        if n_sampling > 0:
            x_sampling = x_train[index, :]
            y_sampling = y_train[index, :]
            nTest = x_sampling.shape[0]

            text_length_pred = np.zeros((nTest, len(classes)), dtype=int)
            
            for i in range(0, nTest):
                text_length = self._lengthDescription(x_sampling[i, :], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, featureRelevance=self.featureRelevance)
                text_length_pred[i] = np.argsort(text_length)

                for c in np.array((range(0, len(classes)))):
                    if y_sampling[ i, text_length_pred[i, :(c+1) ]  ].sum() > 0:
                        self.count[c:] = self.count[c:] + 1
                        break

            self.rank_position = np.array( range(0, np.where( (self.count/self.nTrainTotal > 0.9) )[0][0] +1) )
        else:
            self.rank_position = np.array([0])

        #print(self.rank_position)
        times = self._compTime(times, 'PosicaoConfiancaRanking', timer_start)

        self.centroidsComb, self.norm_centroidsComb = self._calcCentroids(self.nTrainLP, self.frequencyLP, idFeature = None)        
        self.classes = classes
        self.nFeatures = nFeatures

        return self#, times





    def partial_fit(self, x_train, y_train, classes=None, classes_multiclasse=None):
        """
        Train the classifier
        """
        self.fit(x_train, y_train, classes, classes_multiclasse=self.classesMeta)
       




    def predict(self, x_test):
        print('omega', self.omega, 'omegaComb', self.omegaComb )
        """
        Predict
        """
        times={}
        times.update({'Similaridade': 0})
        times.update({'K': 0})
        times.update({'Prob': 0})
        times.update({'L': 0})
        times.update({'Dependencia': 0})

        
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csc_matrix(x_test)

        alpha = 0.001 # parametro usado no calculo to tamanho da descricao para o DFS, Gain e outros
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]

        y_pred = lil_matrix(  np.zeros( (nTest, len(self.classes)), dtype=int )  )

        # Buscando a quantidade de classes Método MetaC - Tang
        numClass = (np.round( self.clfClasses.predict(x_test) )).astype(int) #np.ones( (nTest), dtype=int )## classificador que determina a quantidade de rótulos de cada exemplo

        # Calculando as Centroides das classes
        timer_start = time.time()
        times = self._compTime(times, 'Similaridade', timer_start)

        for i in range( nTest ):
            
            # Calculo tamanho de Descrição
            timer_start = time.time()
            text_length = self._lengthDescription(x_test[i,:], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha = alpha, featureRelevance = self.featureRelevance, times = times)
            times = self._compTime(times, 'L', timer_start)
            
            # Dependencia baseada nas ocorrencias entre as classes
            #if numClass[i] > 1:
            #ranking = sorted(list(enumerate(text_length)), key=lambda lt: lt[1])
            #text_length = text_length + ranking[0][1] * ((1-self.MatCorrelacaoClasses[ranking[0][0]]/self.nTrain[ranking[0][0]]))
                        
            ranking_dep = sorted(range(len(text_length)), key=text_length.__getitem__)

            # Correlação Baseada nas combinações
            if len(text_length) > 0:
                timer_start = time.time()

                classes = []

                posClass = numClass[i]
                    
                classInterval = np.array( range(((-1)*posClass), ((len(self.classesIntervalo)-posClass)+1) ) )
                
                gaussian = self._gaussianFunction(classInterval, 0, self.sd[ posClass ])
                nClass = classInterval[ np.round(gaussian,5)>0.0 ] + posClass


                # Seleciona as combinações de acordo com o numero de classes e as classes mais provaveis do ranking
                try:
                    index =  ( (self.lc.sum(axis=1) >= nClass.min()) & (self.lc.sum(axis=1) <= nClass.max()) ) & (  (self.lc[:,ranking_dep[:len(self.rank_position)] ]==1).any(axis=1) )  #(self.lc.sum(axis=1)==m) &  ( ((self.lc.sum(axis=1) >= nClass.min()) & (self.lc.sum(axis=1) <= nClass.max()) ).all(axis=0) )         
                except:
                    index = []

                classes = list( self.lc[index])
                #for c in self.lc[index]:
                #    classesI.append(list( sp.sparse.find( c )[1]   ))

                if (len(classes) > 1):
                    # seleciona os dados das combinações e calcula o tamanho da descrição para cada uma delas
                    nTrain, frequency, freqTokensClass, binary_frequency = self.nTrainLP[index], self.frequencyLP[:,index], self.freqTokensClassLP[index], self.binary_frequencyLP[:,index]#self._calcFrequencias(self.frequencyLP, self.binary_frequencyLP, self.nTrainLP, self.lc, classes)

                    lengthTextoDep = self._lengthDescription(x_test[i,:], nTrain, frequency, self.binary_freq, freqTokensClass, self.omegaComb, self.centroidsComb[:,index], self.norm_centroidsComb[index], alpha, featureRelevance=self.featureRelevance, times=times)

                    penalidadeGauss = 1
                    if len(classes) > 0:
                        nclasses = np.sum( classes, axis=1) - numClass[i]
                        gaussian = self._gaussianFunction(nclasses, 0, self.sd[posClass])
                        penalidadeGauss += 0.2*(1-gaussian)

                    lengthTextoDep = lengthTextoDep * (penalidadeGauss)
                    times = self._compTime(times, 'Dependencia', timer_start)

                    ranking_dep = sorted(range(len(lengthTextoDep)), key=lengthTextoDep.__getitem__)
                    y_pred[i] = classes[ranking_dep[0]]

                elif (len(classes) == 1):
                    y_pred[i, :] = self.lc[index][0]
                else:
                    y_pred[i, ranking_dep[:numClass[i]]] = 1
            else:
                y_pred[i, ranking_dep[:numClass[i]]] = 1

        return y_pred #, times


    def predict_proba(self, x_test):
        """
        Predict
        """
                
        if not sp.sparse.issparse(x_test):
            x_test = sp.sparse.csc_matrix(x_test)

        alpha = 0.001 # parametro usado no calculo to tamanho da descricao para o DFS, Gain e outros
        nFeatures = x_test.shape[1]
        nTest = x_test.shape[0]
    
        # Buscando a quantidade de classes Método MetaC - Tang
        numClass = self.clfClasses.predict(x_test) # classificador que determina a quantidade de rótulos de cada exemplo

        text_length_pred = np.zeros( (nTest, len(self.classes)), dtype=int )

        for i in range( nTest ):
            
            text_length = self._lengthDescription(x_test[i,:], self.nTrain, self.frequency, self.binary_freq, self.freqTokensClass, self.omega, self.centroids, self.norm_centroids, alpha = alpha, featureRelevance = self.featureRelevance, times = None)
            
            ValorRanking = text_length
            #if numClass[i] > 1:
            #    ranking = sorted(list(enumerate(ValorRanking)), key=lambda lt: lt[1])
            #    ValorRanking = text_length + ranking[0][1] * ((1-self.MatCorrelacaoClasses[ranking[0][0]]/self.nTrain[ranking[0][0]]))
            
            #ValorRanking = np.nan_to_num(ValorRanking)
            text_length_pred[i] = list((-1)*ValorRanking)
        
        return np.array(text_length_pred)

    # Funções do método
    # Calcula a relevância dos termos
    def _calc_feature_relevance(self, binary_frequency, nTrain=None):

        if 'CF' in self.relevance_method:
            featureRelevance = fs.fatorConfidencia(binary_frequency)
        elif 'DFS' in self.relevance_method:
            featureRelevance = fs.calcula_dfs(binary_frequency, nTrain)
        else:
            featureRelevance = np.zeros(binary_frequency.shape[0])

        return featureRelevance

    # Calcula o Tamanho de descrição
    def _lengthDescription(self, x_test, nTrain, frequency, binary_frequency, freqTokensClass, omega, centroids, norm_centroids, alpha = 0.001, featureRelevance=None, times=None):

        idFeature = sp.sparse.find( x_test )[1]#returns the non-negative values of a sparse matrix
        
        if len(idFeature):
            freqTokensClass[freqTokensClass == 0] = float('inf')

            # Relevância dos termos
            timer_start = time.time()
            if featureRelevance is None:
                featureRelevance = self._calc_feature_relevance(binary_frequency[idFeature], nTrain=nTrain)
            else:
                featureRelevance = featureRelevance[idFeature]
            
            k_relevancy = 1/(1+alpha-(featureRelevance))   #np.log2
            times = self._compTime(times, 'K', timer_start)

            # Probabilidade
            timer_start = time.time()
            probToken =  ( ( (frequency[idFeature]+(1/omega))/(freqTokensClass+1) )  ).T 

            tokenLength = np.ceil( -np.log2(probToken) )
            times = self._compTime(times, 'Prob', timer_start)

            # Similaridade Cosseno
            timer_start = time.time()
            norm_doc = sp.sparse.linalg.norm( x_test[:,idFeature] )
            
            cosine_similarity = np.where(( norm_doc * norm_centroids) == 0, 0,  ( x_test[:,idFeature] * centroids[idFeature] ) / ( norm_doc * norm_centroids ))[0]
            times = self._compTime(times, 'Similaridade', timer_start)

            text_length = np.sum( tokenLength * ( k_relevancy ), axis=1 ) * ( -np.log2(0.5*(cosine_similarity)) ) #1/(     np.power( cosine_similarity ,1) )#( -np.log2(0.5*(cosine_similarity + alpha)) ) #

            
            
        else:
            tokenLength = 0
            k_relevancy = 0
            cosine_similarity = 0

            text_length = np.where(sum(nTrain) == 0, 0, nTrain/sum(nTrain))
            text_length = -np.log2( text_length ) #probabilidade da classe




        return text_length

    # Centroids
    def _calcCentroids(self, nTrain, frequency, idFeature=None):

        centroids = np.nan_to_num( np.where(nTrain == 0, 0, frequency / nTrain) )
        
        if idFeature is None:
            norm_centroids = np.linalg.norm(centroids, axis=0)
        else:
            norm_centroids = np.linalg.norm(centroids[idFeature], axis=0)

        return centroids, norm_centroids

    # Aumenta Tamanho da Matriz para o treinamento Online
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

    # Calcula a Função Gaussiana
    def _gaussianFunction(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # Computa o tempo
    def _compTime(self, times, column, startTime):
        if times is not None:
            times.update({column: times[column] + ( time.time() - startTime )})
        return times




    #def _calcFrequencias(self, frequencyLP, binary_frequencyLP, nTrainLP, combClasses, classes, best=None):

    #    nTrain = np.zeros( len(classes) )
    #    frequency = np.zeros( (frequencyLP.shape[0],len(classes)) )    
    #    freqTokensClass = np.zeros( len(classes) )
    #    binary_frequency = np.zeros( (frequencyLP.shape[0],len(classes)) )    

    #    MatCorrelacaoClasses = np.zeros( ( len(classes) ) )   #np.zeros( ( len(classes), len(classes) ) )

    #    for i, v in enumerate(classes):
    #        if (best is not None and v == best):
    #            idClasse =  (combClasses.sum(axis=1)==len(v)) & (combClasses[:,v]==1).all(axis=1) #i é o número da classe
    #        else:
    #            if type(v) is list or type(v) is np.ndarray:
    #                idClasse =  (combClasses.sum(axis=1)==len(v)) & (combClasses[:,v]==1).all(axis=1) #i é o número da classe
    #            else:
    #                idClasse = (combClasses.sum(axis=1)==len(v)) & (combClasses[:,v]==1) #i é o número da classe

    #        nTrain[i] += nTrainLP[idClasse].sum() #conta o número de exemplos de treinamento para cada classe
            #idClasse = np.where(y_train==i)#i é o número da classe
            
    #        frequency[:,i] += np.asarray(frequencyLP[:, idClasse].sum(axis=1))
            
    #        freqTokensClass[i] = frequency[:,i].sum() #soma as ocorrencias de todos os tokens na classe atual
    #        binary_frequency[:,i] += binary_frequencyLP[:, idClasse].sum(axis=1)


            #for k in range(0, self.lc.shape[1]):
            #    if k not in v:   #if k not in v:
            #        g = v.copy()
            #        g.append(k)#[i, k]
            #    else:
            #        g=i
                    
            #    if type(g) is list or type(g) is np.ndarray:
            #        idClasseLP = (frequencyLP.sum(axis=0)>0) & (combClasses[:,g]==1).all(axis=1) #i é o número da classe
            #    else:
            #        idClasseLP = (frequencyLP.sum(axis=0)>0) & (combClasses[:,g]==1) #i é o número da classe
                

            #    MatCorrelacaoClasses[i, k] += nTrainLP[idClasseLP].sum()#spearman(y_train[:,i], y_train[:,j])[0]
                

    #    return nTrain, frequency, freqTokensClass, binary_frequency, nTrain#MatCorrelacaoClasses

if __name__ == "__main__":

    mdl = MDLText(clfClasses=skl.linear_model.SGDClassifier(random_state=5))

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
    