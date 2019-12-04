# -*- coding: utf-8 -*- 
import numpy as np
import sklearn as skl
import skmultilearn as skml
import re #regular expression
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from pandas import read_csv

#plt.switch_backend('agg')
import multiprocessing
from multiprocessing import Process
import time, timeit

#import skmultiflow as skmf
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import neural_network
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import multiclass
from sklearn import model_selection
from sklearn import metrics

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer #english
from nltk.stem import RSLPStemmer #portuguese
from nltk.stem import snowball #other languages

import sys
sys.path.append('utils')
import scipy
import myFunctions
from myFunctions import imprimiResultados_multilabel
import stratified_multilabel_KFold

import load_dataset
from load_dataset import import_dataset_multilabel

sys.path.append('MDLText')
from MDLText_multilabel_MD import MDLText as MDLText_MD
from MDLText_multilabel_MD_V2 import MDLText as MDLText_MD_V2

import configDefault as de

import heapq
import queue as Q

def main():
    """
    Função principal
    """ 
    learning = 'Online'

    # Verificando se o script foi chamado com algum argumento para a execução do experimento
    argDefaut = [None, None, None, None]
    for i, arg in enumerate(sys.argv[1:]):
        print(arg)
        if arg is not None:
            argDefaut[i] = [''+arg+'']


    # Nome da base de dados utilizada para identificar o experimento no arquivo de resultados que será gerado pelo algoritmo    
    # Estrutura: Nome da base de dados, tipo e quantidade de Rótulos (o ultimo é obrigatório para base de dados Mulan)
    datasets = de.getDefautDatasets(argDefaut[0])
    print(datasets)

    performExperimentTypes = de.getPerformExperimentTypes(learning, argDefaut[1])
    print(performExperimentTypes)

    # Forma de atualização do focabulário e de treinamento incremental
    vocabularyTypes = de.getVocabularies(learning, argDefaut[2])
    print(vocabularyTypes)

    feedbacks = de.getFeedbacks(learning, argDefaut[3])
    print(feedbacks)
     
    
    for vocab in vocabularyTypes:

        # Vocabulário Fixo e Dinâmico
        dynamicVocabulary = not(vocab == 'Fixo')

        for performExperimentType in performExperimentTypes:
            
            # Configuração para a execução dos Experimentos em aprendizado Online
            # n_executions: 
            #    Possíveis valores: Int
            #	       Número de execuções do experimento
            # perc_train: 
            #    Possíveis valores: Int
            #	       Porcentagem de Amostras para o treinamento
            # performGrid: 
            #    Possíveis valores: True, False
            #	       True: usa grid search (busca em grade) nos métodos (e.g. SVM, KNN e Random Forest) 
            #                que são sensíveis a variação de parâmetros.
            #	       False: não usa grid search em nenhum método 
            # n_splitsGrid: 
            #    Possíveis valores: Int
            #	       Número de iterações da validação cruzada do grid search
                
            perc_train = 20
            n_executions = 5
            performGrid  = False
            n_splitsGrid = 5

            for datasetName, datasetType, n_datasetLabels, ref, language in datasets:
                print(datasetName)

                # função usada para importar a base de dados. 
                # Essa função retorna as seguintes variáveis:
                #      dataset: um array de 1 coluna, onde cada linha corresponde a uma mensagem 
                #      classes: um vetor com as classes do problema
                #      target: um vetor com as classes de cada mensagem contida no array "dataset"
                dataset, classes, target = import_dataset_multilabel(datasetName, datasetType, n_datasetLabels, n_executions*2)

                # Separando uma porcentagem para Validação 
                datasetTest = None
                targetTest = None

                # Separação estratificada da base de dados multilabel
                cv = stratified_multilabel_KFold.stratified_multilabel_KFold(n_splits = int(100/perc_train), shuffle = True, randomSeed=1)
                splits = cv.split(target)
                if n_executions < splits.shape[0]:
                    splits = splits[:n_executions]

                # Executa o mesmo experimento para cada Feedback
                for feedback in feedbacks:
                
                    # endereço do arquivo onde os resultados da classificação serão guardados.
                    pathResults = de.getDefautPathResults(learning, vocab, performExperimentType)+'resultsOnline_'+performExperimentType+'_'+ datasetName+'_feedback'+feedback +'_media.csv'

                    # lista com os métodos já executados:
                    methodsExecuted = None#verify_methods_executed( pathResults )

                    # lista com os métodos que serão executados:
                    methods = de.getDefautMethods(learning, vocab, methodsExecuted=methodsExecuted)          
        
                    vetWait=[]
                    if feedback != 'Imediato':
                        # gerados as possíveis esperas aleatórias    
                        np.random.seed(1)
                        vetWait.append(np.random.randint(20, size = dataset.shape[0]))
                        vetWait.append(np.random.choice([True, False], size = dataset.shape[0]))

                    # Para cada método da lista de métodos, executa um experimento com os parâmetros informados
                    for methodName, termWeighting in methods:

                        # imprimi o nome do método que será executado nessa iteração
                        print('\n\n\n########################################')
                        print('%s' %(methodName)) 
                        print('########################################\n')

                        # termWeighting: usado para indicar qual esquema de pesos dos termos
                        #     Possíveis valores: 'TF', 'binary', 'TFIDF_sklearn', 'TFIDF'
                        #          'binary': os pesos dos termos são 0 se o termo aparece no texto ou 1 caso não apareça
                        #          'TFIDF': TFIDF calculado por meio da função apresentada no artigo "MDLText: An efficient and lightweight text classifier"

                        # executa um experimento com o método da iteração atual
                        result = perform_experiment_online(dataset, target, classes, methodName, datasetName, pathResults, performGrid, termWeighting, n_splitsGrid, feedback=feedback, dynamicVocabulary=dynamicVocabulary, language=language, vetWait= vetWait, splits=splits, datasetTest=datasetTest, targetTest=targetTest)
                        

def perform_experiment_online(dataset, target, classes, methodName, datasetName, pathResults, performGrid, termWeighting, n_splitsGrid, feedback='Imediato', dynamicVocabulary = False, language=None, vetWait=None, splits=None, datasetTest=None, targetTest=None):
      
    """
    Função usada para executar os experimentos

    Parameters:
    -----------
    dataset: 
        Um array de 1 coluna, onde cada linha corresponde a uma mensagem ou, no caso do Mulan, uma linha contendo os atributos
    
    target:
        Um vetor com as classes de cada mensagem contida no array "dataset"

    classes:
        Um vetor com as classes do dataset

    methodName: string
        Um nome usado para identificar o método. Caso deseje, acrescente outros métodos dentro da função return_classifier(). 

    datasetName: string
        Nome da base de dados que será usado para identificar o experimento no arquivo de resultados que será gerado pelo algoritmo
        
    pathResults: string
        Endereço do arquivo onde você deseja que os resultados da classificação sejam guardados.
        Se o arquivo indicado não existir, ele será criado. Caso já exista, os resultados serão acrescentados ao fim do arquivo.
        
    performGrid: boolean
    	       True: usa grid search (busca em grade) nos métodos (e.g. SVM, KNN e Random Forest) que são sensíveis a variação de parâmetros.
    	       False: não usa grid search em nenhum método  

    termWeighting: string
        Usado para indicar qual esquema de pesos você quer usar para os termos
        Possíveis valores: 'TF', 'binary', 'TFIDF_sklearn', 'TFIDF'
              'TF': term frequency 
              'binary': os pesos dos termos são 0 se o termo aparece no texto ou 1 caso não apareça
              'TFIDF_sklearn': TFIDF calculado por meio da função do scikit learn
              'TFIDF': TFIDF calculado por meio da função apresentada no artigo "MDLText: An efficient and lightweight text classifier"

    n_splitsGrid: int
        Usado para indicar o numero de divisões para o gridSearch
                
    """

    # possiveis classes da base de dados
    classesDataset = classes 
    
    # cria uma lista vazia para guardar os resultados obtidos nas execuções
    results = []  
    resultsLearningCurve = []  

    ## Separação estratificada da base de dados multilabel # perc_train
    #cv = stratified_multilabel_KFold.stratified_multilabel_KFold(n_splits = int(100/perc_train), shuffle = True)
    #splits = cv.split(target)

    #for i in range(n_executions):
    i=0
    for test_index, train_index in splits:

        startTime = []
        endTime = []

        startTime.append(time.time())


        t_out = False # verifica se nao estourou o tempo de execução do fold, de 2 dias

        train_index = list(np.random.RandomState(i).permutation(train_index))
        test_index = list(np.random.RandomState(i).permutation(test_index))

        print('\n\t==============================================')
        print('\tFeedback: ' + feedback)
        print('\tDataset: ' + datasetName + ' Método: ' + methodName)
        print('\tVocabulário Dinâmico? ' + str(dynamicVocabulary))
        print('\tNº execução: %d' %(i+1))
        print('\t==============================================')

        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]

        x_train, vocabulary, df_train, nDocs = convertTrain(dataset_train, dynamicVocabulary, termWeighting, language = language)

        # Contabiliza a quantidade de dados de treinamento
        print('\tAmostras de treinamento: %d' %((nDocs)))
            
        # Chama a função para retornar um classificador baseado no nome fornecido como parâmetro
        classifier = return_classifier_online(methodName, performGrid, n_splitsGrid, dynamicVocabulary)
        
        # treina o classificador com os dados de treinameto
        if 'MDLText_M' in methodName:
            classifier.fit(x_train, y_train, classes_multiclasse=np.unique(target.sum(axis=1))) 
        else:
            classifier.fit(x_train, y_train) 

        # Se for uma instancia GridSearch do sklearn, é necessário obter apenas o estimador para o teste online
        if isinstance(classifier, skl.model_selection.GridSearchCV):
            classifier = classifier.best_estimator_
        
        endTime.append(time.time())

        startTime.append(time.time())

        # Classes de Predição das Amostras de teste
        y_pred = None

        # Contador de amostras de testes para auxiliar a manipular a prioridade no aprendizado
        c_testSamples = 0
        
        # Amostras que serão aplicadas no treinamento incremental
        incrementalLearningSamples = []

        # Heap que controla a ordem com a qual as amostras são apontadas para o classificador
        heapq.heapify(incrementalLearningSamples)

        # Iniciando a fase de testes e de treinamento incremental
        for idx in range((dataset_test.shape[0])):
            row = dataset_test[[idx]]
            
            # Acrecenta o contador de amostras
            c_testSamples += 1

            x_test = convertTest(dataset_test[[idx]], termWeighting, vocabulary=vocabulary, language=language, df_train=df_train, nDocs=nDocs)
                
            # classifica os dados de teste
            y_pred_test = classifier.predict(x_test) #.astype(int)

            # Alguns classificadores retornam como predição uma matriz\array esparsa.
            # Para padronizar a manipulação, ela precisa ser convertida para array
            if scipy.sparse.issparse(y_pred_test):
                y_pred_test = y_pred_test.toarray()

            if y_pred is None:
                y_pred = y_pred_test
            else:
                y_pred = np.concatenate((y_pred, y_pred_test))

            
            # Caso o classificador nao tenha acertado a predição, é preciso informar como feedback
            if (y_test[idx,:] ^ y_pred_test[0,:]).sum() != 0:

                # Feedback Imediato - Quando ocorre um erro, o classificador logo ja conhece a amostra
                if feedback == 'Imediato':
                    partial_fit = True
                    wait = 0

                # Feedback Incerto - Quando ocorre um erro, não se tem certeza se o classificador irá conhecer a amostra 
                elif feedback == 'Incerto':
                    partial_fit = vetWait[1][(test_index[idx])] # Espera aleatória fixa
                    wait = 0

                # Feedback com possibilidade de atraso - Quando ocorre um erro, o classificador 
                # poderá conhecer a amostra com um atraso
                elif feedback == 'Atraso':
                    partial_fit = True
                    #wait = np.random.randint(20)
                    wait = vetWait[0][(test_index[idx])] # Espera aleatória fixa

                # Feedback Incerto e com possibilidade de atraso - Quando ocorre um erro, o classificador
                # poderá ou não conhecer a amostra e poderá conter atraso
                else:
                    partial_fit = vetWait[1][(test_index[idx])] # Espera aleatória fixa
                    wait = vetWait[0][(test_index[idx])] # Espera aleatória fixa

                # As amostras que entrarão no aprendizado Incremental será colocada em uma estrutura heap
                # amostras com Feedback imediato ou com atraso menor serão apresentadas ao classificador primeiro
                if partial_fit:
                    heapq.heappush(incrementalLearningSamples,[(c_testSamples+wait), idx])

            # Aprendizado Incremental            
            while (len(incrementalLearningSamples) > 0 and
                  incrementalLearningSamples[0][0] <= c_testSamples) :

                # Obtém a amostra com maior prioridade
                waitTime, j = heapq.heappop(incrementalLearningSamples)

                # Acrescenta o número de documentos de treinamento e a quantidade de documentos em que cada termo aparece
                x_train, vocabulary, df_train, nDocs = convertTrain(dataset_test[[j]], dynamicVocabulary, termWeighting, vocabulary = vocabulary, language = language, df_train = df_train, nDocs = nDocs)
                            
                # Apresenta esta amostra ao classificador para um treinamento incremental
                classifier.partial_fit(x_train, y_test[[j]])



        endTime.append( time.time() ) # Finaliza o timer
        print('%i' %(endTime[1] - startTime[0]) )

        if (endTime[1] - startTime[0]) > 172800:
            t_out = True
            break

        # Chama a função 'inf_teste' para calcular e retornar o desempenho da classificação. 
        # Essa função calcula a acurácia, F-medida, Precisão e várias outras medidas.
        auxResults = myFunctions.inf_teste_multilabel(y_test, y_pred, classes, startTime=startTime, endTime=endTime)  
        
        # Adiciona os resultados da execução atual na lista de resultados
        results.append( auxResults ) 
        i+=1

    print('Médias')

    # a função 'imprimiResultados' salva os resultados da classificação em formato CSV.
    # Se o arquivo indicado pela variável 'pathResults' não existir, ele será criado. Caso já exista, os resultados serão acrescentados ao fim do arquivo.
    if t_out:
        myFunctions.imprimiMedias_multilabel(None,None,pathResults,methodName,datasetName, printResults=True, timeout=(endTime[1] - startTime[0]))
        myFunctions.imprimiResultados_multilabel(None,pathResults.replace('_media', ''),methodName,datasetName, timeout=(endTime[1] - startTime[0]))
    else:
        myFunctions.imprimiMedias_multilabel(results,classesDataset,pathResults,methodName,datasetName, printResults=True)
        myFunctions.imprimiResultados_multilabel(results,pathResults.replace('_media', ''),methodName,datasetName)
        
    return resultsLearningCurve

def return_classifier_online(method, performGrid, n_splitsGrid, metaclassificadorOnline = False):
    """
    Função usada para selecionar um método de classificação para ser usado no experimento
 
    Parameters:
    -----------
    method: string
        Um nome usado para identificar o método. Caso deseje, acrescente outros métodos dentro da função. 
        
    performGrid: boolean
  	    True: usa grid search (busca em grade) nos métodos (e.g. SVM, KNN e Random Forest) que são sensíveis a variação de parâmetros.
   	    False: não usa grid search em nenhum método  

    n_splitsGrid: boolean
  	    int: indica o valor de k na validação cruzada
    """ 

    param_grid = None
    # Métodos de Transformação de Problema 
    if ('1vsAll' in method or '1vsRest' in method or 'TP-CC' in method ): #('TP-BR' in method or 'TP-LP' in method or 'TP-CC' in method or '1vsAll' in method):

        if 'M.NB' in method: #multinomial naive Bayes
            # Inicia o classificador com os parâmetros default do Scikit
            classifier = skl.naive_bayes.MultinomialNB()

        elif 'B.NB' in method: #bernoulli naive Bayes
            # Inicia o classificador com os parâmetros default do Scikit
            classifier = skl.naive_bayes.BernoulliNB()

        elif 'Naive_Bayes' in method: #bernoulli naive Bayes
            # Inicia o classificador com os parâmetros default do Scikit
            classifier = NaiveBayes()#NaiveBayes()

        elif 'Perceptron' in method: #perceptron
            # inicia o classificador com os parâmetros default do Scikit
            classifier = skl.linear_model.Perceptron()

            #if performGrid:
            #    param_grid = {'estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 100, 1000]}#,
                              #'penalty': [None, "l2","l1","elasticnet"]}
                              #'solver' : ['lbfgs', 'sgd', 'adam']} 

        elif 'SGD' in method: #SGD
            # inicia o classificador com os parâmetros default do Scikit
            classifier = skl.linear_model.SGDClassifier(random_state=5)

        elif 'PA' in method: #Passivo-Agressivo
            # inicia o classificador com os parâmetros default do Scikit
            classifier = skl.linear_model.PassiveAggressiveClassifier(random_state=5)
        
        elif 'MDLText' in method: #Passivo-Agressivo
            # inicia o classificador com os parâmetros default do Scikit
            classifier = MDLText()

        if '1vsAll' in method:
            # inicia o classificador de transformação binária
            #classifier = skml.problem_transform.BinaryRelevance( classifier = classifier )  
            classifier = skl.multiclass.OneVsRestClassifier( classifier )

        if 'TP-CC' in method:
            # inicia o classificador de transformação binária
            #classifier = skml.problem_transform.BinaryRelevance( classifier = classifier )  
            classifier = skmf.meta.ClassifierChain( base_estimator = classifier )

		# Adiciona a opção de busca em grade, caso seja necessario
        if param_grid is not None:
            classifier = skl.model_selection.GridSearchCV(classifier, cv=n_splitsGrid, param_grid=param_grid, scoring = 'f1_macro')
        
    elif method == 'MLP':
        # inicia o classificador com os parâmetros default do Scikit
        classifier = skl.neural_network.MLPClassifier(random_state=5)

        #if performGrid:
        #    param_grid = {'hidden_layer_sizes': [[1], [100], [1000], [10000]],
        #                  'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 100, 1000],
        #                  'activation': ['identity', 'logistic', 'tanh', 'relu']}
        #                  'solver' : ['lbfgs', 'sgd', 'adam']} 

    elif 'MDLText' in method: #MDLText

        metaclass = skl.linear_model.SGDClassifier(random_state=5)

        if '_MD_V2' in method: #MDLText Versão meta-Classificador que considera a dependencia entre as classes
            classifier  = MDLText_MD_V2(clfClasses = metaclass, calc_feature_relevance=False)

        elif '_MD' in method: #MDLText Versão meta-Classificador que considera a dependencia entre as classes
            classifier  = MDLText_MD(metaclass = metaclass)

    return classifier



def convertTrain(dataset_train, dynamicVocabulary, termWeighting, vocabulary=None, language=None, df_train = None, nDocs=None):

    if ((df_train is None and vocabulary is not None ) or
        (df_train is not None and vocabulary is None )):
        print('Erro')
        exit

    # Se a matriz for esparsa, quer dizer que é uma base de dados já convertida do Mulan
    if scipy.sparse.issparse(dataset_train): 

        # Inicia o vocabulário e as variáveis necessarias para o treinamento incremental
        if vocabulary is None:
            vocabulary = (dataset_train.sum(axis=0)).nonzero()[1]
            df_train = np.zeros((1, len(vocabulary)), dtype=int)

        # Carrega os dados de Treinamento
        x_train = dataset_train[:,vocabulary]

        # Atribuindo a frequência e o numero de documentos de treinamento           
        df_train += (x_train != 0).sum(axis=0)

        if nDocs is None:
            nDocs = 0  
        nDocs += x_train.shape[0]                        

    else:
        # Converte o texto para TF utilizando um criando vocabulario inicial
        if (vocabulary is None):
            # Gera o vetor termo-frequência para os textos
            vectorizer = skl.feature_extraction.text.CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, lowercase = True, binary=False, dtype=np.int32)
            x = vectorizer.fit_transform(dataset_train)
            vocabulary = vectorizer.vocabulary_
        
        # Acrescenta o número de documentos de treinamento e a quantidade de documentos em que cada termo aparece
        # Converte para TF novamente, mas dessa vez é preciso acrescentar novos termos ao vocabulário se houver
        x_train, vocabulary, df_train, nDocs = myFunctions.text2tf(dataset_train.copy(), add_vocabulary = dynamicVocabulary, vocabulary = vocabulary, language=language, feature_frequency = df_train, nDocs = nDocs, add_Docs = True)
        
    # Converte a representação TF para TF-IDF  
    if termWeighting == 'TFIDF':  
        x_train = myFunctions.tf2tfidf(x_train, df=df_train, nDocs = nDocs, normalize_tf=True, normalize_tfidf=True)
        
    # Converte a representação TF para binária
    elif termWeighting == 'binary':
        x_train[x_train!=0]=1 # convert os dados para representação binária

    return x_train, vocabulary, df_train, nDocs


def convertTest(dataset_test, termWeighting, vocabulary, language, df_train, nDocs):
        
    if scipy.sparse.issparse(dataset_test):
        # Carrega a amostra de teste
        x_test = dataset_test[:,vocabulary]
    else:
        # Converte o texto da amostra de treinamento para termo-frequência
        x_test = scipy.sparse.csc_matrix( myFunctions.text2tf(dataset_test[:], vocabulary=vocabulary, language=language, feature_frequency=df_train) )
    
    # Converte a representação TF para TF-IDF  
    if termWeighting == 'TFIDF':  
        x_test = myFunctions.tf2tfidf(x_test, df=df_train, nDocs=nDocs, normalize_tf=True, normalize_tfidf=True)

    # Converte a representação TF para binária
    elif termWeighting == 'binary':
        x_test[x_test!=0]=1 #convert os dados para representação binária

    return x_test

def perform_fold_with_limited_time( func, args, timeWait):
    """Runs a function with time limit
    :param func: The function to run
    :param timeWait: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    p = Process(target=func, 
                args=args, 
                kwargs={})

    startTime = time.time() #finaliza o timer
    p.start()
    p.join(timeWait)
    if p.is_alive():
        p.terminate()
        return True, (time.time()-startTime)

    return False, None

if __name__ == "__main__":
    
    main() #executa a função principal   


	 



