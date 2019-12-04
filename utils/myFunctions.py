# -*- coding: utf-8 -*- 

import numpy as np
import sklearn as skl
import scipy
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy import stats
from scipy.stats import chi2


from collections import Mapping, defaultdict
import matplotlib.pyplot as plt

from sklearn import preprocessing

import pandas as pd
import re #regular expression
import os
import sys 
import array
import math

import nltk
from nltk import word_tokenize

import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import collections
from collections import Counter, defaultdict


import skmultilearn as skml
from skmultilearn import problem_transform

#==================================================
#Função para calcular as métricas de classificação
#==================================================
def inf_teste(matriz_confusao, classes, printResults=True, freq_classes=None, startTime=[0,0], endTime=[0,0]):
  """
  Função usada calcular as medidas de desempenho da classificação.
  """
  #print(matriz_confusao)
  
  n_teste = sum(sum(matriz_confusao))
  
  nClasses = len( matriz_confusao ) #numero de classes
  vp=np.zeros( (1,nClasses) )
  vn=np.zeros( (1,nClasses) )
  fp=np.zeros( (1,nClasses) )
  fn=np.zeros( (1,nClasses) )

  #Laço para encontrar vp, vn, fp e fn de todas as classes
  for i in range(0,nClasses):
    vp[0,i] = matriz_confusao[i,i];
    fn[0,i] = sum(matriz_confusao[i,:])-matriz_confusao[i,i];
    fp[0,i] = sum(matriz_confusao[:,i])-matriz_confusao[i,i];
    vn[0,i] = n_teste - vp[0,i] - fp[0,i] - fn[0,i];

  sensitividade = vp/(vp+fn) #recall
  sensitividade = np.nan_to_num(sensitividade) #Replace nan with zero and inf with finite numbers

  fpr = fp/(fp+vn) #false positive rate 
  fpr = np.nan_to_num(fpr) #Replace nan with zero and inf with finite numbers
  
  especificidade = vn/(fp+vn) #especificidade
  #acuracia = (vp+vn)/(vp+vn+fp+fn)
  acuracia = np.zeros( (1,nClasses ) ) #inicializa a variavel acuracia
  acuracia[0:nClasses]=np.sum(vp)/(vp[0,0]+vn[0,0]+fp[0,0]+fn[0,0])#quantidade de acertos dividido pelo numero de testes
  
  precisao = vp/(vp+fp); #precision
  precisao = np.nan_to_num(precisao) #Replace nan with zero and inf with finite numbers
  
  f_medida = (2*precisao*sensitividade)/(precisao+sensitividade)
  f_medida = np.nan_to_num(f_medida) #Replace nan with zero and inf with finite numbers
  
  mcc = ( (vp*vn)-(fp*fn) ) / np.sqrt( (vp+fp)*(vp+fn)*(vn+fp)*(vn+fn) )

  #microAverage average
  sensitividade_microAverage = sum(vp[0,:])/sum(vp[0,:]+fn[0,:])#sensitividade ou recall
  precisao_microAverage = sum(vp[0,:])/sum(vp[0,:]+fp[0,:])
  f_medida_microAverage = (2*precisao_microAverage*sensitividade_microAverage)/(precisao_microAverage+sensitividade_microAverage)

  #macro average
  auxSensitividade_macroAverage = vp[0,:]/(vp[0,:]+fn[0,:])
  auxSensitividade_macroAverage = np.nan_to_num(auxSensitividade_macroAverage) #Replace nan with zero and inf with finite numbers
  sensitividade_macroAverage = sum( auxSensitividade_macroAverage )/nClasses#sensitividade ou recall

  auxPrecisao_macroAverage = vp[0,:]/(vp[0,:]+fp[0,:])
  auxPrecisao_macroAverage = np.nan_to_num(auxPrecisao_macroAverage) #Replace nan with zero and inf with finite numbers
  precisao_macroAverage = sum( auxPrecisao_macroAverage )/nClasses

  f_medida_macroAverage = (2*precisao_macroAverage*sensitividade_macroAverage)/(precisao_macroAverage+sensitividade_macroAverage);

  #coeficiente Kappa
  sumLinhas = np.zeros( (1,nClasses) )
  sumColunas = np.zeros( (1,nClasses) )
  for i in range(0,nClasses):
    sumLinhas[0,i] = sum(matriz_confusao[i,:]);
    sumColunas[0,i] = sum(matriz_confusao[:,i]);

  rand = sum( (sumLinhas[0,:]/n_teste)*(sumColunas[0,:]/n_teste) )
  kappa_coefficient = (acuracia[0][0] - rand)/(1-rand);
  
  if printResults == True:
      print('\n\tFPR        Recall     Espec.   Precisao   F-medida   Classe/Freq')
      for i in range(0,nClasses):
        print('\t%1.3f      %1.3f      %1.3f    %1.3f      %1.3f      %s/%s' % (fpr[0,i], sensitividade[0,i], especificidade[0,i], precisao[0,i], f_medida[0,i],classes[i], (freq_classes[i] if freq_classes is not None else 0) ) )
    
      print('\t---------------------------------------------------------------------');
      #imprimi as médias
      print('\t%1.3f      %1.3f      %1.3f    %1.3f      %1.3f      Media' % (np.mean(fpr), np.mean(sensitividade), np.mean(especificidade), np.mean(precisao), np.mean(f_medida) ) )
      print('\t.....      %1.3f      .....    %1.3f      %1.3f      Macro-Average' % (sensitividade_macroAverage, precisao_macroAverage, f_medida_macroAverage) )
      print('\t.....      %1.3f      .....    %1.3f      %1.3f      Micro-Average\n' % (sensitividade_microAverage, precisao_microAverage, f_medida_microAverage) )
    
      print('\tacuracia: %1.3f' %acuracia[0,0])
      print('\tkappa_coefficient:  %1.3f' %kappa_coefficient)
      if nClasses==2:
          print('\tMCC:  %1.3f' %mcc[0,0])

  resultados = {'fpr': fpr, 'sensitividade': sensitividade, 'especificidade': especificidade, 'acuracia': acuracia, 'precisao':precisao, 'f_medida':f_medida, 'mcc':mcc}
  resultados.update({'SMa':sensitividade_macroAverage, 'PMa':precisao_macroAverage, 'FMMa':f_medida_macroAverage})
  resultados.update({'SMi':sensitividade_microAverage, 'PMi':precisao_microAverage, 'FMMi':f_medida_microAverage})
  resultados.update({'kappa_coefficient': kappa_coefficient})
  resultados.update({'confusionMatrix': matriz_confusao})
  resultados.update({'timeFolds': endTime[1] - startTime[0]})
  
  return resultados #return like a dictionary

#==================================================
#Função para calcular as métricas de classificação
#==================================================
def inf_teste_multilabel(y_true, y_pred, classes = None, printResults=True, startTime=0, endTime=0, pred_decision=None):
    nClasses = y_true.shape[1]
    nAmostras = y_true.shape[0]
        
    if classes is None:
        classes = list(range(nClasses))
        
    if scipy.sparse.issparse(y_pred):
        y_pred = y_pred.toarray().astype(int)     


    if scipy.sparse.issparse(pred_decision):
        pred_decision = pred_decision.toarray()


    ############ Informações sobre a predição Número de Classes ############
    y_true_classes_n = y_true.sum(axis=1)
    y_pred_classes_n = y_pred.sum(axis=1)
    

    target_classes_n = y_true.sum(axis=1)
    classes_n = np.unique( target_classes_n )
    freq_classes_n = np.zeros(len(classes_n), dtype=int)
    for i in target_classes_n:
        index = np.where(classes_n==i)
        
        if len(index)>0:
            freq_classes_n[index] += 1

    freq_classes_n = freq_classes_n/nAmostras

    erros = 0
    quantidade_menor = 0
    quantidade_maior = 0
    for i in range( nAmostras ):
        if y_true_classes_n[i] != y_pred_classes_n[i]:#signifca que ele errou o número de labels
            erros += 1; 

            if y_true_classes_n[i] > y_pred_classes_n[i]: # Signifca que ele apontou um número de labels maior
                quantidade_menor += 1
            else: # Signifca que ele apontou um número de labels menor
                quantidade_maior += 1

    erros_total = erros / nAmostras
    erros_quantidade_menor_classe = quantidade_menor / nAmostras
    erros_quantidade_maior_classe = quantidade_maior / nAmostras
    ########################################################

    # Analisando a capacidade de Predição do numero de Classes
    cm = skl.metrics.confusion_matrix(y_true_classes_n, y_pred_classes_n, classes_n)
    auxResults = inf_teste(cm, classes_n, printResults=False, freq_classes=freq_classes_n)
    
    FMMa_nClasse = auxResults['FMMa']
    FMMi_nClasse = auxResults['FMMi']
    FMMe_nClasse = np.mean(auxResults['f_medida'])


    ###########################
    hamming_loss = 0.0
    for i in range( y_true.shape[0] ):
        auxXor = y_true[i,:] ^ y_pred[i,:]  #xor operation
        hamming_loss += (auxXor.sum()/nClasses) 
        
    hamming_loss = hamming_loss * 1/y_true.shape[0]
    ###########################
    
    ###########################
    acertos = 0
    for i in range( y_true.shape[0] ):
        auxXor = y_true[i,:] ^ y_pred[i,:]  #xor operation
        if auxXor.sum() == 0: #signifca que ele acertou todos os labels
            acertos += 1; 
    subset_accuracy = acertos/y_true.shape[0] #subset_accuracy - tambem conhecida como Exact match accuracy
    ###########################

    jaccard_index = skl.metrics.jaccard_similarity_score(y_true, y_pred)  #jaccard_similarity_score
       
    ###########################################################################################


    if pred_decision is not None:
        
        ###################### One-Error ######################
        one_error = 0
        error = 0
        for i in range( y_true.shape[0] ):
            ranking = sorted(list(enumerate(pred_decision[i])), reverse=True, key=lambda lt: lt[1])

            if y_true[i,ranking[0][0]] != 1:#signifca que ele acertou a primeira posição do Ranking
                error += 1

        one_error = error / y_true.shape[0]
        ########################################################
    
        #################### Ranking-Error #####################
        ranking_loss_index = skl.metrics.label_ranking_loss(y_true, pred_decision)  # Ranking_error
        ########################################################

        #################### Average Precision-Error #####################
        average_precision_error_index = skl.metrics.label_ranking_average_precision_score(y_true, pred_decision)  # Average_precision_error
        ########################################################
    
        #################### Coverage-Error ####################
        coverage_error_index = skl.metrics.coverage_error(y_true, pred_decision)  # Coverage_error
        ########################################################
    else:
        one_error = float('Nan')
        ranking_loss_index = float('Nan')
        average_precision_error_index = float('Nan')
        coverage_error_index = float('Nan')
        

    #################### Accuracy-Index ####################
    accuracy_index = 0
    for i in range( y_true.shape[0] ):
        auxAnd = y_true[i,:] & y_pred[i,:]  #and operation
        auxOr  = y_true[i,:] | y_pred[i,:]  #or operation

        accuracy_index = accuracy_index + auxAnd.sum()/auxOr.sum()    
    accuracy_index = accuracy_index / y_true.shape[0]   #accuracy_score       
    ###########################################################################################

    #precisao, f-medida, sensitividade
    vp=np.zeros( (1,nClasses) )
    vn=np.zeros( (1,nClasses) )
    fp=np.zeros( (1,nClasses) )
    fn=np.zeros( (1,nClasses) )
    for i in range( nClasses ):
        matriz_confusao = skl.metrics.confusion_matrix(y_true[:,i], y_pred[:,i], labels=[0,1])
        #print(matriz_confusao)
        idxPosClass = 1
        vp[0,i] = matriz_confusao[idxPosClass,idxPosClass];
        fn[0,i] = sum(matriz_confusao[idxPosClass,:])-matriz_confusao[idxPosClass,idxPosClass];
        fp[0,i] = sum(matriz_confusao[:,idxPosClass])-matriz_confusao[idxPosClass,idxPosClass];
        vn[0,i] = y_true.shape[0] - vp[0,i] - fp[0,i] - fn[0,i];        

    sensitividade = vp/(vp+fn) #recall
    sensitividade = np.nan_to_num(sensitividade) #Replace nan with zero and inf with finite numbers
    
    fpr = fp/(fp+vn) #false positive rate 
    fpr = np.nan_to_num(fpr) #Replace nan with zero and inf with finite numbers
      
    especificidade = vn/(fp+vn) #especificidade
    #acuracia = (vp+vn)/(vp+vn+fp+fn)

    acuracia=(vp+vn)/( y_true.shape[0] )#quantidade de acertos individual para cada classe
      
    precisao = vp/(vp+fp); #precision
    precisao = np.nan_to_num(precisao) #Replace nan with zero and inf with finite numbers
      
    f_medida = (2*precisao*sensitividade)/(precisao+sensitividade)
    f_medida = np.nan_to_num(f_medida) #Replace nan with zero and inf with finite numbers


    #microAverage average
    sensitividade_microAverage = sum(vp[0,:])/(sum(vp[0,:])+sum(fn[0,:]))#sensitividade ou recall
    precisao_microAverage = sum(vp[0,:])/(sum(vp[0,:])+sum(fp[0,:]))
    f_medida_microAverage = (2*precisao_microAverage*sensitividade_microAverage)/(precisao_microAverage+sensitividade_microAverage)

    #macro average
    sensitividade_macroAverage = np.mean(np.nan_to_num( vp[0,:]/(vp[0,:]+fn[0,:]) ) ) #sensitividade ou recall #Replace nan with zero and inf with finite numbers
    precisao_macroAverage = np.mean(np.nan_to_num( vp[0,:]/(vp[0,:]+fp[0,:]) ) ) #Replace nan with zero and inf with finite numbers
    f_medida_macroAverage = np.mean(f_medida)#(2*precisao_macroAverage*sensitividade_macroAverage)/(precisao_macroAverage+sensitividade_macroAverage) # Forma errada de se calcular a Macro average multirrótulo


    #################### Precision-Index ####################
    precisaoGeral = 0
    for i in range( y_true.shape[0] ):
        auxAnd = y_true[i,:] & y_pred[i,:]  #and operation

        precisaoGeral = precisaoGeral + np.nan_to_num((auxAnd.sum())/(y_pred[i,:].sum()))
    precisaoGeral = precisaoGeral / (y_true.shape[0])   #Precision_score       
    prec2 = vp/(vp+fp); 
    ##########################################################

    #################### Recall-Index ####################
    sensitividadeGeral = 0
    for i in range( y_true.shape[0] ):
        auxAnd = y_true[i,:] & y_pred[i,:]  #and operation

        sensitividadeGeral = sensitividadeGeral + np.nan_to_num((auxAnd.sum())/(y_true[i,:].sum()))
    sensitividadeGeral = sensitividadeGeral / (y_true.shape[0])   #Recall_score       
    ##########################################################

    #################### Fmedida-Index ####################
    f_medidaGeral = 0
    for i in range( y_true.shape[0] ):
        auxAnd = y_true[i,:] & y_pred[i,:]  #and operation

        f_medidaGeral = f_medidaGeral + 2*(auxAnd.sum()) / ((y_pred[i,:].sum()) + (y_true[i,:].sum()))
    f_medidaGeral = f_medidaGeral / (y_true.shape[0])   #Recall_score       
    ##########################################################

    ###########################################################################################3

    if printResults == True:
      print('F Medida: %1.3f' %np.mean(f_medidaGeral) )
      print('F1 macro: %1.3f' %f_medida_macroAverage )
      print('F1 micro: %1.3f' %f_medida_microAverage )

      print('\nAcuracia   FPR        Recall     Espec.   Precisao   F-medida         Classe')
      for i in range(0,nClasses):
        print('%1.3f      %1.3f      %1.3f      %1.3f    %1.3f      %1.3f       %s' % (acuracia[0,i], fpr[0,i], sensitividade[0,i], especificidade[0,i], precisao[0,i], f_medida[0,i], classes[i] ) )
    
      print('------------------------------------------------------------------------------------');
      #imprimi as médias
      print('%1.3f      %1.3f      %1.3f      %1.3f    %1.3f      %1.3f       Media' % (np.mean(acuracia), np.mean(fpr), np.mean(sensitividade), np.mean(especificidade), np.mean(precisao), np.mean(f_medida) ) )
      print('.....      .....      %1.3f      .....    %1.3f      %1.3f       Macro-Average' % (sensitividade_macroAverage, precisao_macroAverage, f_medida_macroAverage) )
      print('.....      .....      %1.3f      .....    %1.3f      %1.3f       Micro-Average\n' % (sensitividade_microAverage, precisao_microAverage, f_medida_microAverage) )
    
      print('hammingLoss: %1.3f' %hamming_loss)  
      print('Subset Accuracy: %1.3f' %subset_accuracy )
      print('accuracy_index: %1.3f' %accuracy_index ) #accuracy_score   
      print('coverage_error: %1.3f' %coverage_error_index) 
      
    if printResults:   
        print('\nScikit results')
        print('F1 macro: %1.3f' %skl.metrics.f1_score(y_true, y_pred, average='macro') )
        print('F1 micro: %1.3f' %skl.metrics.f1_score(y_true, y_pred, average='micro') )
        print('hammingLoss: %1.3f' %skl.metrics.hamming_loss(y_true, y_pred) )  
        print('Subset Accuracy: %1.3f' %skl.metrics.accuracy_score(y_true, y_pred) )
        print('jaccard_index: %1.3f' %skl.metrics.jaccard_similarity_score(y_true, y_pred) )  #jaccard_similarity_score ) #jaccard_similarity_score    
        print('zero_one_loss: %1.3f' %skl.metrics.zero_one_loss(y_true, y_pred) ) #igual a 1-accuracy_score
        print('coverage_error: %1.3f' %skl.metrics.coverage_error(y_true, y_pred) )
        print('label_ranking_average_precision_score: %1.3f' %skl.metrics.label_ranking_average_precision_score(y_true, y_pred) )
        print('label_ranking_loss: %1.3f' %skl.metrics.label_ranking_loss(y_true, y_pred) )
        
    resultados = {}
    resultados.update({'PMa': precisao_macroAverage})
    resultados.update({'PMi': precisao_microAverage})
    resultados.update({'SMa': sensitividade_macroAverage})
    resultados.update({'SMi': sensitividade_microAverage})
    resultados.update({'FMMa': f_medida_macroAverage})
    resultados.update({'FMMi': f_medida_microAverage})
    resultados.update({'HL': hamming_loss})
    resultados.update({'SAcc': subset_accuracy})
    resultados.update({'jaccard_index': jaccard_index})
    resultados.update({'Acc': accuracy_index})
    resultados.update({'CE': coverage_error_index})
    resultados.update({'RL': ranking_loss_index})
    resultados.update({'Tempo': endTime[1] - startTime[0]})
    resultados.update({'Erro_n_classe': erros_total})
    resultados.update({'Erro_n_classe_inferior': erros_quantidade_menor_classe})
    resultados.update({'Erro_n_classe_superior': erros_quantidade_maior_classe})
    resultados.update({'FMMa_n_classe': FMMa_nClasse})
    resultados.update({'FMMi_n_classe': FMMi_nClasse})
    resultados.update({'FMMe_n_classe': FMMe_nClasse})
    resultados.update({'One_error': one_error})
    resultados.update({'Average_precision': average_precision_error_index})
    resultados.update({'PMe': precisaoGeral})
    resultados.update({'SMe': sensitividadeGeral})
    resultados.update({'FMMe': f_medidaGeral})
    resultados.update({'TempoTreino': endTime[0] - startTime[0]})
    resultados.update({'TempoTeste': endTime[1] - startTime[1]})
    
    return resultados #return like a dictionary

#============================================
#Função para imprimir as médias dos folds
#============================================
def imprimiMedias(resultados, classes, pathResults = None, methodName = None, datasetName = None, printResults = False):

    acuracia = np.zeros( (len(resultados),len(classes)) )
    fpr = np.zeros( (len(resultados),len(classes)) )
    sensitividade = np.zeros( (len(resultados),len(classes)) )
    especificidade = np.zeros( (len(resultados),len(classes)) )
    precisao = np.zeros( (len(resultados),len(classes)) )
    f_medida = np.zeros( (len(resultados),len(classes)) )
    mcc = np.zeros( (len(resultados),len(classes)) )

    sensitividade_macroAverage = np.zeros( (1,len(resultados)) )
    precisao_macroAverage = np.zeros( (1,len(resultados)) )
    f_medida_macroAverage = np.zeros( (1,len(resultados)) )
    f_medida_macroAverage_sklearn = np.zeros( (1,len(resultados)) )

    sensitividade_microAverage = np.zeros( (1,len(resultados)) )
    precisao_microAverage = np.zeros( (1,len(resultados)) )
    f_medida_microAverage = np.zeros( (1,len(resultados)) )

    f_medida_microAverage = np.zeros( (1,len(resultados)) )

    kappa_coefficient = np.zeros( (1,len(resultados)) )
    roc_auc = np.zeros( (1,len(resultados)) )
    timeFolds = np.zeros( (1,len(resultados)) )

    for i in range(0,len(resultados)):
        acuracia[i,:] = resultados[i]['acuracia']
        fpr[i,:] = resultados[i]['fpr']
        sensitividade[i,:] = resultados[i]['sensitividade']
        especificidade[i,:] = resultados[i]['especificidade']
        precisao[i,:] = resultados[i]['precisao']
        f_medida[i,:] = resultados[i]['f_medida']
        mcc[i,:] = resultados[i]['mcc']

        sensitividade_macroAverage[0,i] = resultados[i]['SMa']
        precisao_macroAverage[0,i] = resultados[i]['PMa']
        f_medida_macroAverage[0,i] = resultados[i]['FMMa']
        f_medida_macroAverage_sklearn[0,i] = resultados[i]['FMMa_sklearn']

        sensitividade_microAverage[0,i] = resultados[i]['SMi']
        precisao_microAverage[0,i] = resultados[i]['PMi']
        f_medida_microAverage[0,i] = resultados[i]['FMMi']
        
        kappa_coefficient[0,i] = resultados[i]['kappa_coefficient']
        if 'roc_auc' in resultados[i]:
            roc_auc[0,i] = resultados[i]['roc_auc']
        else:
            roc_auc[0,i] = 0.0
        timeFolds[0,i] = resultados[i]['timeFolds']

    if printResults:     
        print('\tFPR        Recall     Espec.   Precisao   F-medida   Classe');
        for i in range(0,len(classes)):
            print('\t%1.3f      %1.3f      %1.3f      %1.3f    %1.3f      %s' % (np.mean(fpr[:,i]), np.mean(sensitividade[:,i]), np.mean(especificidade[:,i]), np.mean(precisao[:,i]), np.mean(f_medida[:,i]), classes[i] ) )

        print('\t------------------------------------------------------------------------------------');
        print('\t%1.3f      %1.3f      %1.3f      %1.3f    %1.3f      Media' % (np.mean(fpr), np.mean(sensitividade), np.mean(especificidade), np.mean(precisao), np.mean(f_medida) ) )
        print('\t.....      %1.3f      .....      %1.3f    %1.3f      Macro-Average' % (np.mean(sensitividade_macroAverage[0,:]), np.mean(precisao_macroAverage[0,:]), np.mean(f_medida_macroAverage[0,:])) )
        print('\t.....      %1.3f      .....      %1.3f    %1.3f      Micro-Average' % (np.mean(sensitividade_microAverage[0,:]), np.mean(precisao_microAverage[0,:]), np.mean(f_medida_microAverage[0,:])) )

        print('\n\tAcuracia: %1.3f' % np.mean(acuracia))
        print('\tkappa_coefficient: %1.3f' % np.mean(kappa_coefficient[0,:]))
        print('\troc_auc: %1.3f' % np.mean(roc_auc[0,:]))
        print('\ttimeFolds: %1.3f' % np.mean(timeFolds[0,:]))

    if pathResults is not None:         
        os.makedirs(os.path.dirname(pathResults), exist_ok = True)
        if not os.path.isfile(pathResults):#se o arquivo não existe, adiciona os labels das colunas do .csv
            fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos
            fileWrite.write('base_dados,metodo,acuracia,fpr,sensitividade,especificidade,precisao,F-medida,mcc,sensitividadeMacro,precisaoMacro,F-medidaMacro,sensitividadeMicro,precisaoMicro,F-medidaMicro,kappa,roc_auc,tempo,F-medidaMacro_SKLearn')    
            fileWrite.close();
            
        fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos
        
        fileWrite.write('\n%-30s,%-30s,%1.3f,%1.3f,' %(datasetName, methodName, np.mean(acuracia),  np.mean(fpr)   ))
        fileWrite.write('%1.3f,' %( np.mean(sensitividade[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(especificidade[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(precisao[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(f_medida[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(mcc[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(sensitividade_macroAverage[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(precisao_macroAverage[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(f_medida_macroAverage[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(sensitividade_microAverage[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(precisao_microAverage[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(f_medida_microAverage[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(kappa_coefficient[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(roc_auc[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(timeFolds[0,:]) ))
        fileWrite.write('%1.3f,' %( np.mean(f_medida_macroAverage_sklearn[0,:]) ))

        ##xmlLabels = labels_to_xml(classes)
        #xmlConfusionMatrix = matrix_to_xml( resultados[i]['confusionMatrix'] )
        #fileWrite.write('%s%s' %( xmlLabels,xmlConfusionMatrix ))	

    
        fileWrite.close();
    

#============================================
#Função para imprimir as médias dos folds
#============================================
def imprimiMedias_multilabel(resultados, classes, pathResults = None, methodName = None, datasetName = None, printResults = False, timeout=None):

    if resultados is not None:

        sensitividade_Average = np.zeros( (1,len(resultados)) )
        precisao_Average = np.zeros( (1,len(resultados)) )
        f_medida_Average = np.zeros( (1,len(resultados)) )

        sensitividade_macroAverage = np.zeros( (1,len(resultados)) )
        precisao_macroAverage = np.zeros( (1,len(resultados)) )
        f_medida_macroAverage = np.zeros( (1,len(resultados)) )

        sensitividade_microAverage = np.zeros( (1,len(resultados)) )
        precisao_microAverage = np.zeros( (1,len(resultados)) )
        f_medida_microAverage = np.zeros( (1,len(resultados)) )

        hammingLoss = np.zeros( (1,len(resultados)) )
        subset_accuracy = np.zeros( (1,len(resultados)) ) #tambem conhecida como Exact match accuracy
        jaccard_index = np.zeros( (1,len(resultados)) ) 
        accuracy_index = np.zeros( (1,len(resultados)) ) 
        coverage_error_index = np.zeros( (1,len(resultados)) ) 
        rankingLoss = np.zeros( (1,len(resultados)) )
        timeFolds = np.zeros( (1,len(resultados)) )
        timeTrain = np.zeros( (1,len(resultados)) )
        timeTest = np.zeros( (1,len(resultados)) )

        erro_n_classeAverage = np.zeros( (1,len(resultados)) )
        erro_n_classe_inferiorAverage = np.zeros( (1,len(resultados)) )
        erro_n_classe_superiorAverage = np.zeros( (1,len(resultados)) )

        FMMa_n_classeAverage = np.zeros( (1,len(resultados)) )
        FMMi_n_classeAverage = np.zeros( (1,len(resultados)) )
        FMMe_n_classeAverage = np.zeros( (1,len(resultados)) )

        One_errorAverage = np.zeros( (1,len(resultados)) )
        Coverage_errorAverage = np.zeros( (1,len(resultados)) )
        Ranking_lossAverage = np.zeros( (1,len(resultados)) )
        Average_precisionAverage = np.zeros( (1,len(resultados)) )

        for i in range(0,len(resultados)):
            sensitividade_macroAverage[0,i] = resultados[i]['SMa']
            precisao_macroAverage[0,i] = resultados[i]['PMa']
            f_medida_macroAverage[0,i] = resultados[i]['FMMa']

            sensitividade_microAverage[0,i] = resultados[i]['SMi']
            precisao_microAverage[0,i] = resultados[i]['PMi']
            f_medida_microAverage[0,i] = resultados[i]['FMMi']
            
            hammingLoss[0,i] = resultados[i]['HL']
            subset_accuracy[0,i] = resultados[i]['SAcc']
            jaccard_index[0,i] = resultados[i]['jaccard_index']
            accuracy_index[0,i] = resultados[i]['Acc']
            coverage_error_index[0,i] = resultados[i]['CE']
            rankingLoss[0,i] = resultados[i]['RL']

            if 'Tempo' in resultados[i]:
                timeFolds[0,i] = resultados[i]['Tempo']
            else:
                timeFolds[0,i] = 0.0

            if 'TempoTreino' in resultados[i]:
                timeTrain[0,i] = resultados[i]['TempoTreino']
            else:
                timeTrain[0,i] = 0.0

            if 'TempoTeste' in resultados[i]:
                timeTest[0,i] = resultados[i]['TempoTeste']
            else:
                timeTest[0,i] = 0.0

            erro_n_classeAverage[0,i] = resultados[i]['Erro_n_classe']
            erro_n_classe_inferiorAverage[0,i] = resultados[i]['Erro_n_classe_inferior']
            erro_n_classe_superiorAverage[0,i] = resultados[i]['Erro_n_classe_superior']

            FMMa_n_classeAverage[0,i] = resultados[i]['FMMa_n_classe']
            FMMi_n_classeAverage[0,i] = resultados[i]['FMMi_n_classe']
            FMMe_n_classeAverage[0,i] = resultados[i]['FMMe_n_classe']

            One_errorAverage[0,i] = resultados[i]['One_error']
            Average_precisionAverage[0,i] = resultados[i]['Average_precision']

            sensitividade_Average[0,i] = resultados[i]['SMe']
            precisao_Average[0,i] = resultados[i]['PMe']
            f_medida_Average[0,i] = resultados[i]['FMMe']

        if printResults:     
            print('f1_macro: %1.3f' % np.mean(f_medida_macroAverage[0,:]))
            print('f1_micro: %1.3f' % np.mean(f_medida_microAverage[0,:]))
            print('hammingLoss: %1.3f' % np.mean(hammingLoss[0,:]))
            print('subset_accuracy: %1.3f' % np.mean(subset_accuracy[0,:]))
            #print('jaccard_index: %1.3f' % np.mean(jaccard_index[0,:]))
            print('accuracy_index: %1.3f' % np.mean(accuracy_index[0,:]))
            print('coverage_error_index: %1.3f' % np.mean(coverage_error_index[0,:]))
            print('rankingLoss: %1.3f' % np.mean(rankingLoss[0,:]))

        if pathResults is not None:         
            os.makedirs(os.path.dirname(pathResults), exist_ok = True)
            if not os.path.isfile(pathResults):#se o arquivo não existe, adiciona os labels das colunas do .csv
                fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos
                fileWrite.write('base_dados,metodo,F-medidaMacro,F-medidaMicro,SensitividadeMacro,SensitividadeMicro,PrecisaoMacro,PrecisaoMicro,hammingLoss,subset_accuracy,accuracy_index,coverage_error,rankingLoss,tempo,Erro_n_classe,Erro_n_classe_inferior,Erro_n_classe_superior,FMMa_n_classe,FMMi_n_classe,FMMe_n_classe,One_error,Average_precision,F-medida,Sensitividade,Precisao,TempoTreino,TempoTeste')    
                fileWrite.close();

                
            fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos

            fileWrite.write('\n%-30s,%-30s,%1.3f,' %(datasetName, methodName, np.mean(f_medida_macroAverage[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(f_medida_microAverage[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(sensitividade_macroAverage[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(sensitividade_microAverage[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(precisao_macroAverage[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(precisao_microAverage[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(hammingLoss[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(subset_accuracy[0,:]) ))
            #fileWrite.write('%1.3f,' %( np.mean(jaccard_index[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(accuracy_index[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(coverage_error_index[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(rankingLoss[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(timeFolds[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(erro_n_classeAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(erro_n_classe_inferiorAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(erro_n_classe_superiorAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(FMMa_n_classeAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(FMMi_n_classeAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(FMMe_n_classeAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(One_errorAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(Average_precisionAverage[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(f_medida_Average[0,:])  ))   
            fileWrite.write('%1.3f,' %( np.mean(sensitividade_Average[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(precisao_Average[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(timeTrain[0,:])  ))
            fileWrite.write('%1.3f' %( np.mean(timeTest[0,:])  ))

            fileWrite.close();
    else:

        if printResults:     
            print('A execução do Fold Ultrapassou o tempo limite! %1.3f' %(timeout))

        if pathResults is not None:         
            os.makedirs(os.path.dirname(pathResults), exist_ok = True)
            if not os.path.isfile(pathResults):#se o arquivo não existe, adiciona os labels das colunas do .csv
                fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos
                fileWrite.write('base_dados,metodo,F-medidaMacro,F-medidaMicro,SensitividadeMacro,SensitividadeMicro,PrecisaoMacro,PrecisaoMicro,hammingLoss,subset_accuracy,accuracy_index,coverage_error,rankingLoss,tempo,Erro_n_classe,Erro_n_classe_inferior,Erro_n_classe_superior,FMMa_n_classe,FMMi_n_classe,FMMe_n_classe,One_error,Average_precision,F-medida,Sensitividade,Precisao,TempoTreino,TempoTeste')    
                fileWrite.close();
                
            fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos

            fileWrite.write('\n%-30s,%-30s,%-30s,' %(datasetName, methodName, 'T_Out' ))
            fileWrite.write('T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,%1.3f,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out' %(timeout))

            fileWrite.close();

         
#============================================
#Função para imprimir as médias dos folds
#============================================
def imprimiResultados(resultados,classes,end_resultados,metodo,nomeDataset,print_class=None):
    nfolds = len(resultados)

    os.makedirs(os.path.dirname(end_resultados), exist_ok = True)
    if not os.path.isfile(end_resultados):#se o arquivo não existe, adiciona os labels das colunas do .csv
        fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
        fileWrite.write('base_dados,metodo,acuracia,fpr,sensitividade,especificidade,precisao,F-medida,mcc,sensitividadeMacro,precisaoMacro,F-medidaMacro,sensitividadeMicro,precisaoMicro,F-medidaMicro,kappa,roc_auc,tempo,F-medidaMacro,confusionMatrix')    
        fileWrite.close();
        
    fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
    
    for i in range(0,len(resultados)):
        if print_class is None: #imprimi a média
            fileWrite.write('\n%-30s,%-20s,%1.3f,%1.3f,' %( nomeDataset, metodo, np.mean(resultados[i]['acuracia']), np.mean(resultados[i]['fpr']) ))
            fileWrite.write('%1.3f,%1.3f,%1.3f,' %( np.mean(resultados[i]['sensitividade']), np.mean(resultados[i]['especificidade']), np.mean(resultados[i]['precisao']) )) 
            fileWrite.write('%1.3f,%1.3f,' %( np.mean(resultados[i]['f_medida']), np.mean(resultados[i]['mcc']) ))
            
        else: 
            idClass = classes.index( print_class )
            fileWrite.write('\n%-30s,%-20s,%1.3f,%1.3f,' %( nomeDataset, metodo, resultados[i]['acuracia'][0,idClass], resultados[i]['fpr'][0,idClass] ))
            fileWrite.write('%1.3f,%1.3f,%1.3f,' %( resultados[i]['sensitividade'][0,idClass], resultados[i]['especificidade'][0,idClass], resultados[i]['precisao'][0,idClass] )) 
            fileWrite.write('%1.3f,%1.3f,' %( resultados[i]['f_medida'][0,idClass], resultados[i]['mcc'][0,idClass] ))

            
        fileWrite.write('%1.3f,%1.3f,%1.3f,' %( resultados[i]['SMa'],  resultados[i]['PMa'], resultados[i]['FMMa'] ))
        fileWrite.write('%1.3f,%1.3f,%1.3f,' %( resultados[i]['SMi'],  resultados[i]['PMi'], resultados[i]['FMMi'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['kappa_coefficient'] ))
        if 'roc_auc' in resultados[i]:
            roc_auc = resultados[i]['roc_auc']
        else:
            roc_auc = 0.0
        fileWrite.write('%1.3f,' %( roc_auc ))
        
        if 'Tempo' in resultados[i]:
            timeFolds = resultados[i]['Tempo']
        else:
            timeFolds = 0.0
        fileWrite.write('%1.3f,' %( timeFolds ))	

        fileWrite.write('%1.3f,' %( resultados[i]['FMMa_sklearn'] ))

        xmlLabels = labels_to_xml(classes);
        xmlConfusionMatrix = matrix_to_xml( resultados[i]['confusionMatrix'] );
        fileWrite.write('%s%s' %( xmlLabels,xmlConfusionMatrix ))	
        
    fileWrite.close();
    
def imprimiResultados_multilabel(resultados,end_resultados,metodo,nomeDataset, timeout=None):

    if resultados is not None:

        nfolds = len(resultados)

        os.makedirs(os.path.dirname(end_resultados), exist_ok = True)
        if not os.path.isfile(end_resultados):#se o arquivo não existe, adiciona os labels das colunas do .csv
            fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
            fileWrite.write('base_dados,metodo,F-medidaMacro,F-medidaMicro,SensitividadeMacro,SensitividadeMicro,PrecisaoMacro,PrecisaoMicro,hammingLoss,subset_accuracy,accuracy_index,coverage_error,rankingLoss, tempo,Erro_n_classe,Erro_n_classe_inferior,Erro_n_classe_superior,FMMa_n_classe,FMMi_n_classe,FMMe_n_classe,One_error,Average_precision,F-medida,Sensitividade,Precisao')    
            fileWrite.close();
            
        fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
        
        for i in range(0,len(resultados)):          
            

            fileWrite.write('\n%-30s,%-30s,%1.3f,' %(nomeDataset, metodo, resultados[i]['FMMa'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['FMMi'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['SMa'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['SMi'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['PMa'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['PMi'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['HL'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['SAcc'] ))
            #fileWrite.write('%1.3f,' %( resultados[i]['jaccard_index'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['Acc'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['CE'] ))
            fileWrite.write('%1.3f,' %( resultados[i]['RL'] ))
            
            if 'Tempo' in resultados[i]:
                timeFolds = resultados[i]['Tempo']
            else:
                timeFolds = 0.0
            fileWrite.write('%1.3f,' %( timeFolds ))


            fileWrite.write('%1.3f,' %( resultados[i]['Erro_n_classe']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['Erro_n_classe_inferior']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['Erro_n_classe_superior']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['FMMa_n_classe']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['FMMi_n_classe']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['FMMe_n_classe']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['One_error']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['Average_precision']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['FMMe']  ))   
            fileWrite.write('%1.3f,' %( resultados[i]['SMe']  ))
            fileWrite.write('%1.3f,' %( resultados[i]['PMe']  ))

            if 'Tempo' in resultados[i]:
                timeTrain = resultados[i]['TempoTreino']
            else:
                timeTrain = 0.0
            fileWrite.write('%1.3f,' %( timeTrain ))

            if 'Tempo' in resultados[i]:
                timeTest = resultados[i]['TempoTeste']
            else:
                timeTest = 0.0
            fileWrite.write('%1.3f' %( timeTest ))	

        fileWrite.close();
    else:

        os.makedirs(os.path.dirname(end_resultados), exist_ok = True)
        if not os.path.isfile(end_resultados):#se o arquivo não existe, adiciona os labels das colunas do .csv
            fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
            fileWrite.write('base_dados,metodo,F-medidaMacro,F-medidaMicro,SensitividadeMacro,SensitividadeMicro,PrecisaoMacro,PrecisaoMicro,hammingLoss,subset_accuracy,accuracy_index,coverage_error,rankingLoss, tempo,Erro_n_classe,Erro_n_classe_inferior,Erro_n_classe_superior,FMMa_n_classe,FMMi_n_classe,FMMe_n_classe,One_error,Average_precision,F-medida,Sensitividade,Precisao,TempoTreino,TempoTeste')    
            fileWrite.close();
            
        fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
        
        fileWrite.write('\n%-30s,%-30s,%-30s,' %(nomeDataset, metodo, 'T_Out' ))
        fileWrite.write('\nT_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,%1.3f,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out,T_Out' %(timeout))
        fileWrite.close();



def labels_to_xml(labels):
     xmlString = ''
     if (type(labels) is list or
         type(labels) is np.ndarray):
         for i in range( len(labels) ):
            xmlString = xmlString+r'<l>'+str(labels[i])+r'</l>'
     else:
         for i in range( len(labels) ):
            xmlString = xmlString+r'<l>'+labels[i][0]+r'</l>'

                                               
     xmlString = r'<labels>'+xmlString+r'</labels>'
    
     return xmlString
    
def matrix_to_xml(data):
    
    xmlString = '';
    for i in range( data.shape[0] ):
        xmlString = xmlString+r'<row id="'+str(i+1)+r'">';
        for j in range( data.shape[1] ):
            if j==0:
                xmlString = xmlString+str( data[i,j] )
            else:
                xmlString = xmlString+' '+str( data[i,j] )
        xmlString = xmlString+r'</row>'
    xmlString = r'<matrix>'+xmlString+r'</matrix>';
                
    return xmlString

#============================================
#Função separar em treino e teste
#============================================
def crossValidation_holdOut(target, pTrain, shuffleData = 1, classes=None):

    if classes is None:
        #extrai as classes dada a lista de targets #transform numerical labels in non-numerical labels
        #--------------------------------------------
        le = preprocessing.LabelEncoder()
        le.fit(auxTarget)
        classes = list(le.classes_)
        target = le.transform(auxTarget)
        #auxTarget2=le.inverse_transform(target)
        #--------------------------------------------
       
    idTrainClasse = np.zeros( len(classes),dtype='object')
    idTestClasse = np.zeros( len(classes),dtype='object')

    for i in range( 0,len(classes) ):
        idClasse = np.where(target==classes[i])#i é o número da classe
        idClasse = idClasse[0]
        
        np2 = len(idClasse) #conta o número de exemplos de treinamento para cada classe
        sep = int(np.ceil(np2*pTrain))

        if shuffleData==1:
            perm = np.random.permutation(range(np2))
            idClasse = idClasse[perm] #embaralha os dados
            
        #vetores para treinamento
        idTrainClasse[i] = idClasse[0:sep]
        idTestClasse[i] = idClasse[sep:]

    train_index = idTrainClasse[0]
    test_index = idTestClasse[0]

    for i in range( 1,len(classes) ):
        train_index = np.concatenate( (train_index,idTrainClasse[i]) )
        test_index = np.concatenate( (test_index,idTestClasse[i]) )

    if shuffleData==1:
        np2 = len(train_index)
        perm = np.random.permutation(range(np2))
        train_index = train_index[perm]
        
        np2 = len(test_index)
        perm = np.random.permutation(range(np2))  
        test_index = test_index[perm]
    else: #ordena os indices para manter a mesma ordem inicial dos índices de cada classe
        train_index = np.sort(train_index)
        test_index = np.sort(test_index)

    return train_index, test_index



#============================================
#Função separar em treino e teste - mysql
#============================================
def crossValidation_holdOut_mysql(connection, tableName, targetColumn, idColumn, pTrain, shuffleData = 1, classes=None):

    #verifica quais são as possíveis classes
    with connection.cursor() as cursor:        
        cursor.execute('select distinct %s from %s' %(targetColumn, tableName)) 
         
        auxClasses = cursor.fetchall()

    classes = []
    for row in auxClasses:
        classes.append( row[targetColumn] )
           
    
    idTrainClasse = []
    idTestClasse = []

    for i in range( 0,len(classes) ):

        with connection.cursor() as cursor: 
            cursor.execute('select ' + idColumn + ' from ' + tableName + ' where Category = %s', (classes[i]))
            
            auxidClasse = cursor.fetchall()
         
        idClasse = np.zeros( len(auxidClasse),dtype='object')  
        for i in range ( len(auxidClasse) ):
            idClasse[i] = auxidClasse[i][idColumn]       
        
        np2 = len(idClasse) #conta o número de exemplos de treinamento para cada classe
        sep = int(np.ceil(np2*pTrain))

        if shuffleData==1:
            perm = np.random.permutation(range(np2))
            idClasse = idClasse[perm] #embaralha os dados
            
        #vetores para treinamento
        idTrainClasse.append(idClasse[0:sep])
        idTestClasse.append(idClasse[sep:])

    train_index = idTrainClasse[0]
    test_index = idTestClasse[0]

    for i in range( 1,len(classes) ):
        train_index = np.concatenate( (train_index,idTrainClasse[i]) )
        test_index = np.concatenate( (test_index,idTestClasse[i]) )

    if shuffleData==1:
        np2 = len(train_index)
        perm = np.random.permutation(range(np2))
        train_index = train_index[perm]
        
        np2 = len(test_index)
        perm = np.random.permutation(range(np2))  
        test_index = test_index[perm]
    else: #ordena os indices para manter a mesma ordem inicial dos índices de cada classe
        train_index = np.sort(train_index)
        test_index = np.sort(test_index)

    return train_index, test_index



#============================================
#Função separar em treino e teste
#============================================
def preparaDadosKfolds(target, k, shuffleData = 1,randomSeed = None, classes=None):
    
    if classes is None:
        #extrai as classes dada a lista de targets #transform numerical labels in non-numerical labels
        #--------------------------------------------
        le = preprocessing.LabelEncoder()
        le.fit(auxTarget)
        classes = list(le.classes_)
        target = le.transform(auxTarget)
        #auxTarget2=le.inverse_transform(target)
        #--------------------------------------------
        
    id_folds = np.zeros( (len(classes),k),dtype='object')
    for id in range( 0,len(classes) ):
        idClasse = np.where(target==classes[id])#i é o número da classe
        idClasse = idClasse[0]
        np2 = len(idClasse) #conta o número de exemplos de treinamento para cada classe
        tamanhoSub = int(np.floor(np2/k)) #;%se a divisão for maior que .5 o último fold será maior que os demais, senão será menor
        aux_k = np2-(tamanhoSub*k) #quantidade de folders que terão um "dado" a mais

        if shuffleData==1:
            if randomSeed is None:
                perm = np.random.RandomState().permutation(range(np2))
            else:
                perm = np.random.RandomState(randomSeed).permutation(range(np2))
            idClasse = idClasse[perm] #embaralha os dados

        j1=0
        tamanhoSub+=1 
        for i in range(0,aux_k):
            id_folds[id][i] = idClasse[j1:j1+tamanhoSub]
            j1=j1+tamanhoSub
            
        tamanhoSub-=1;     
        for i in range(aux_k, k):
            id_folds[id][i] = idClasse[j1:j1+tamanhoSub]
            j1=j1+tamanhoSub

    #junta os indices de todas as classes
    folds = np.zeros( k,dtype='object')
    for i in range(0,k):
        folds[i]=id_folds[0][i]
        for id in range( 1,len(classes) ):
            folds[i] = np.concatenate( (folds[i],id_folds[id][i]) )

    #separa os índices de treinamento e de teste
    folds_final = np.zeros( k,dtype='object')
    train_index = np.zeros( k,dtype='object')
    test_index = np.zeros( k,dtype='object')
    for nf in range(0,k):
        #une os outros folds para criar os dados de treinamento
        folds_temp = np.concatenate( (folds[0:nf],folds[nf+1:]) )

        #o fold atual será usado para teste. Os outros folds serão usados para treino
        test_index[nf] = folds[nf]

        train_index[nf] = folds_temp[0]
        for nf2 in range(1,k-1):
            train_index[nf] = np.concatenate( (train_index[nf],folds_temp[nf2]) )

        #embaralha os índices de treinamento e teste
        if shuffleData==1:
            dimTrain = len( train_index[nf] )
            dimTest = len( test_index[nf] )
            
            if randomSeed is None:
                perm_train = np.random.RandomState().permutation(range(dimTrain))
                perm_test = np.random.RandomState().permutation(range(dimTest))
            else:
                perm_train = np.random.RandomState(randomSeed).permutation(range(dimTrain))
                perm_test = np.random.RandomState(randomSeed).permutation(range(dimTest))
                
            train_index[nf] = train_index[nf][perm_train] #embaralha os dados
            test_index[nf] = test_index[nf][perm_test] #embaralha os dados
            
        else: #ordena os indices para manter a mesma ordem inicial dos índices de cada classe
            train_index[nf] = np.sort(train_index[nf])
            test_index[nf] = np.sort(test_index[nf])
            

        folds_final[nf] = np.array( [train_index[nf],test_index[nf]] )

    return folds_final

   
#============================================
#Função separar em treino e teste
#============================================
def random_undersampling(auxTarget, shuffleData = 1, randomSeed = None):

    #extrai as classes dada a lista de targets #transform numerical labels in non-numerical labels
    #--------------------------------------------
    le = preprocessing.LabelEncoder()
    le.fit(auxTarget)
    classes = list(le.classes_)
    target = le.transform(auxTarget)
    #auxTarget2=le.inverse_transform(target)
    #--------------------------------------------
       
    idDadosClasse = np.zeros( len(classes),dtype='object')
    nDados = np.zeros( len(classes), dtype=int )

    idClasse = []
    for i in range( 0,len(classes) ):
        auxIdClasse = np.where(target==i)#i é o número da classe
        idClasse.append( auxIdClasse[0] )
        
        nDados[i] = len(idClasse[i]) #conta o número de exemplos de treinamento para cada classe
        
    
    minClasse = np.min(nDados);
    
    for i in range( 0,len(classes) ):
        if shuffleData==1:
            if randomSeed is None:
                perm = np.random.RandomState().permutation( range(nDados[i]) )
            else:
                perm = np.random.RandomState(randomSeed).permutation( range(nDados[i]) )

            idClasse[i] = idClasse[i][perm] #embaralha os dados
            
        #vetores para treinamento
        idDadosClasse[i] = idClasse[i][0:minClasse]

    dados_index = idDadosClasse[0]
    for i in range( 1,len(classes) ):
        dados_index = np.concatenate( (dados_index,idDadosClasse[i]) )

    if shuffleData==1:
        np2 = len(dados_index)
        
        if randomSeed is None:
            perm = np.random.RandomState().permutation( np2 )
        else:
            perm = np.random.RandomState(randomSeed).permutation( np2 )
        
        dados_index = dados_index[perm]
        
    return dados_index

#============================================
#Função para converter Texto em Vetor Termo-Frequência
#============================================
def text2tf(text, add_vocabulary = False, add_Docs = False, vocabulary = None, language = None, feature_frequency = None, nDocs = None):
    """
    Faz a conversão do texto para um vetors termo-frequencia (TF)
    É possível atualizar o vocabulário e a frequencia dos atributos nos documentos de 
    forma incremental, permitindo que novos documentos sejam analisados em diferentes momentos. 
    """   
    if nDocs is None:
        nDocs = 0

    # Quando for uma nova amostra de treinamento, é necessário adicionar novos termos ao vocabulário
    if add_vocabulary:

        # Inicializa o vocabulario, a frequência de atributos e o número de documentos
        if vocabulary is None:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        if feature_frequency is None:
            feature_frequency = np.zeros((1,1)).astype(int)

        # Separa os termos dos documentos
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
        
        # Adiciona ao vocabulário os termos dos documentos analisados
        for word in tokens:
            try:
                feature_idx = vocabulary[word]
            except KeyError:
                continue

    # Gera o vetor termo-frequência para os textos
    vectorizer = skl.feature_extraction.text.CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, lowercase = True, binary=False, dtype=np.int32, vocabulary = vocabulary )

    tf = vectorizer.transform(text) 

    if add_Docs:

        # Atualiza a frequência dos termos conforme se altera o vocabulário
        # calcula a quantidade de novos termos/atributos para adicionar a frequência
        if feature_frequency is None:
            new_terms = len(vocabulary)
            
        else:
            new_terms = len(vocabulary) - feature_frequency.shape[1]
        new_features = np.zeros( (1,new_terms) ).astype(int)

        # calcula/atualiza a frequência dos atributos
        if feature_frequency is None:
            feature_frequency = (tf != 0).sum(axis=0)
            
        else:
            feature_frequency = np.hstack((feature_frequency,new_features)) + (tf != 0).sum(axis=0)
        nDocs += len(text)

        return tf, vocabulary, feature_frequency, nDocs 
    else:
        return tf

    
#============================================
#Função para converter tf para tf-idf
#============================================
def tf2tfidf(tf, df=None,  nDocs=None, normalize_tf=False, normalize_tfidf=True, return_df=False):
    """
    Faz a conversão para tf_idf. 
    Quando for usado na fase de teste, deve ser passado a frequência de documentos 
    da base de treinamento que contém cada token e a quantidade de documentos de treinamento. 
    
    Uma das diferença dessa função para a função sklearn.feature_extraction.text.TfidfVectorizer 
    é que ela usa log na base 10 em vez do logaritmo natural. Além disso, o Tf é normalizado como 
    np.log10( 1+tf.data ), enquanto no scikit é normalizado como 1 + np.log( tf.data ). Ainda,
    o IDF é calculado como np.log10( (nDocs+1)/(df+1) ), enquanto no scikit é 
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

    #nDocs += 1 #usado para evitar que o idf seja negativo quando o token aparece em todos os documentos         
    
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

#============================================
#Função para coletar o parametro no grid
#============================================
def coleta_parametros_grid(pathFile,datasetProcurado,metodoProcurado,retornar_textFormat='no'):
    
    fileContent = pd.read_csv( pathFile ,sep=',')
    
    metodoProcurado = metodoProcurado.lower(); 
    
    datasetProcurado = datasetProcurado.replace(' ','');
    datasetProcurado = datasetProcurado.replace('_','-');
    datasetProcurado = datasetProcurado.replace('.mat','');
    datasetProcurado = datasetProcurado.replace('.wc','');
    datasetProcurado = datasetProcurado.lower(); #convert para minusculo
    
    fileContent['base_dados'] = fileContent['base_dados'].str.replace(' ','') #replace em todas as linhas da coluna informada
    fileContent['base_dados'] = fileContent['base_dados'].str.replace('_','-');
    fileContent['base_dados'] = fileContent['base_dados'].str.replace('.mat','');
    fileContent['base_dados'] = fileContent['base_dados'].str.replace('.wc','');
    fileContent['base_dados'] = fileContent['base_dados'].str.lower(); #convert para minusculo
    
    fileContent['metodo'] = fileContent['metodo'].str.lower(); #convert para minusculo
      
    fileContent = fileContent.loc[fileContent['base_dados'] == datasetProcurado]  
    fileContent = fileContent.loc[fileContent['metodo'].str.contains('.*'+metodoProcurado+'.*')] 
    
    if len(fileContent) == 0:
        print('Erro! Método não encontrado -- Método: %s -- Dataset: %s' %(metodoProcurado,datasetProcurado))
        sys.exit(0) #interrompe o script
    elif len(fileContent) > 1:
        print('Erro! Mais de uma repetição do método procurado -- Método: %s' %metodoProcurado)
        sys.exit(0) #interrompe o script
    else:
        if retornar_textFormat=='yes':
            if ('tf-idf' in fileContent['metodo'].iloc[0]):
                textFormat = 'tf-idf';
            elif ('tf' in fileContent['metodo'].iloc[0]):
                textFormat = 'tf';
            elif ('binario' in fileContent['metodo'].iloc[0]):
                textFormat = 'binario';
            else:
                print('Erro! Não foi encontrado o formato de texto (TF, TF-IDF ou binario) -- Método procurado: %s' %metodoProcurado)
                sys.exit(0) #interrompe o script
        
        param = None
        if (metodoProcurado=='svm'):
            param = int( fileContent['metodo'].str.replace('(.*_c=2\^)+(-?[0-9]+)+(.*)',r'\2') )
        elif (metodoProcurado=='mdl') or (metodoProcurado=='cf') or (metodoProcurado=='dfs'): 
            param = int( fileContent['metodo'].str.replace('(.*_omega=2\^)+([0-9]+)+(.*)',r'\2') )
        elif (metodoProcurado=='knn'):
            param = int( fileContent['metodo'].str.replace('(.*_k=)+([0-9]+)+(.*)',r'\2') )
        elif (metodoProcurado=='rf'):
            param = int( fileContent['metodo'].str.replace('(.*_ntrees=)+([0-9]+)+(.*)',r'\2') )
            
        if retornar_textFormat=='yes':
            return param, textFormat;
        else:
            return param


def idxWinner_gridSearch( resultados_all, classes, medidaGrid = 'FMMa', print_class=None):
    
    maiorMedida = 0
    idWinner = 0
    for k in range( len(resultados_all) ):
        
        resultados = resultados_all[k]
        
        if medidaGrid=='acuracia' or medidaGrid=='fpr' or medidaGrid=='sensitividade' or medidaGrid=='especificidade' or medidaGrid=='precisao' or medidaGrid=='f_medida' or medidaGrid=='mcc':
            if print_class is None: #imprimi a média 
                aux_medidaDesempenho = np.zeros( (len(resultados),len(classes)) )
        else:
            aux_medidaDesempenho = np.zeros( len(resultados) )
                
                
        for i in range(0,len(resultados)):
                
            if medidaGrid=='acuracia' or medidaGrid=='fpr' or medidaGrid=='sensitividade' or medidaGrid=='especificidade' or medidaGrid=='precisao' or medidaGrid=='f_medida' or medidaGrid=='mcc':
                
                if print_class is None: #imprimi a média                
                    aux_medidaDesempenho[i,:] = resultados[i][medidaGrid]
                    
                else:
                    idClass = classes.index( print_class )
                    aux_medidaDesempenho = resultados[i][medidaGrid][0,idClass]
                    
            else:
                aux_medidaDesempenho[i] = resultados[i][medidaGrid]
        
        medidaDesempenho = np.mean(aux_medidaDesempenho)
        
        if medidaDesempenho > maiorMedida:
            maiorMedida = medidaDesempenho
            idWinner = k
            
    return idWinner, maiorMedida

def analise_medidas(dataset_target, y_true, y_pred, classes = None, printResults=True, startTime=0, endTime=0, pred_decision=None):

    nAmostras = dataset_target.shape[0]
    nAmostrasTeste = y_true.shape[0]
    nClasses = y_true.shape[1]

    resultados = inf_teste_multilabel(y_true, y_pred, classes = classes, printResults=printResults, startTime=startTime, endTime=endTime, pred_decision=pred_decision)

    # Informações para o calculo das Análises de combinações de Classes
    classe_desconhecida = np.ones((nClasses), dtype=int) *(-1)

    comb_classes_possiveis=[]
    for c in np.unique( y_true, axis=0 ):
        comb_classes_possiveis.append(c)
    comb_classes_possiveis.append(classe_desconhecida)

    target_comb_classes = np.zeros((nAmostras), dtype=int)
    for i in range(0,nAmostras):
        try:
            target_comb_classes[i] = [np.array_equal(dataset_target[i],y) for y in comb_classes_possiveis].index(True)
        except:
            target_comb_classes[i] = len(comb_classes_possiveis)-1

    
    freq_comb_classes_possiveis = np.zeros(len(comb_classes_possiveis), dtype=int)
    for i in target_comb_classes:
        freq_comb_classes_possiveis[(i)] += 1
    freq_comb_classes_possiveis = freq_comb_classes_possiveis/nAmostras
    freq_comb_classes_possiveis = np.concatenate((freq_comb_classes_possiveis, [0]), axis=0)

    y_true_comb_classes = np.zeros((nAmostrasTeste), dtype=int)
    for i in range(0,nAmostrasTeste):
        try:
            y_true_comb_classes[i] = [np.array_equal(y_true[i],y) for y in comb_classes_possiveis].index(True)
        except:
            y_true_comb_classes[i] = len(comb_classes_possiveis)-1
    y_true_comb_classes = np.concatenate((y_true_comb_classes, [len(comb_classes_possiveis)-1]), axis=0)

    y_pred_comb_classes = np.zeros((nAmostrasTeste), dtype=int)
    for i in range(0,nAmostrasTeste):
        try:
            y_pred_comb_classes[i] = [np.array_equal(y_pred[i],y) for y in comb_classes_possiveis].index(True)
        except:
            y_pred_comb_classes[i] = len(comb_classes_possiveis)-1
    y_pred_comb_classes = np.concatenate((y_pred_comb_classes, [len(comb_classes_possiveis)-1]), axis=0)


    cm = skl.metrics.confusion_matrix(y_true_comb_classes, y_pred_comb_classes, np.unique(y_true_comb_classes))
    auxResults = inf_teste(cm, comb_classes_possiveis, printResults=True, freq_classes=freq_comb_classes_possiveis)
    
    FMMa_nCombinacao = auxResults['FMMa']
    FMMi_nCombinacao = auxResults['FMMi']
    FMMe_nCombinacao = (auxResults['f_medida'].sum())/auxResults['f_medida'].shape[1]


    resultados.update({'FMMa_n_combinacao': FMMa_nCombinacao})
    resultados.update({'FMMi_n_combinacao': FMMi_nCombinacao})
    resultados.update({'FMMe_n_combinacao': FMMe_nCombinacao})


    return resultados #return like a dictionary


def imprimi_analiseMedidas_multilabel(resultados,end_resultados,metodo,nomeDataset):
    nfolds = len(resultados)

    os.makedirs(os.path.dirname(end_resultados), exist_ok = True)
    if not os.path.isfile(end_resultados):#se o arquivo não existe, adiciona os labels das colunas do .csv
        fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
        fileWrite.write('base de dados,metodo,Erro_n_classe,Erro_n_classe_inferior,Erro_n_classe_superior,FMMa_n_classe,FMMi_n_classe,FMMe_n_classe,FMMa_n_combinacao,FMMi_n_combinacao,FMMe_n_combinacao,One_error,Coverage_error,Ranking_loss,Average_precision')
        fileWrite.close();
        
    fileWrite  = open(end_resultados,"a") #abre arquivo em modo de ediçãos
    
    for i in range(0,len(resultados)):            
        
        fileWrite.write('\n%-30s,%-30s,%1.3f,' %(nomeDataset, metodo, resultados[i]['Erro_n_classe'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['Erro_n_classe_inferior'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['Erro_n_classe_superior'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['FMMa_n_classe'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['FMMi_n_classe'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['FMMe_n_classe'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['FMMa_n_combinacao'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['FMMi_n_combinacao'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['FMMe_n_combinacao'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['One_error'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['CE'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['RL'] ))
        fileWrite.write('%1.3f,' %( resultados[i]['Average_precision'] ))
                            
    fileWrite.close();

#============================================
#Função para imprimir as médias dos folds
#============================================
def imprimi_analiseMedidas_Medias_multilabel(resultados, classes, pathResults = None, methodName = None, datasetName = None):

    erro_n_classeAverage = np.zeros( (1,len(resultados)) )
    erro_n_classe_inferiorAverage = np.zeros( (1,len(resultados)) )
    erro_n_classe_superiorAverage = np.zeros( (1,len(resultados)) )

    FMMa_n_classeAverage = np.zeros( (1,len(resultados)) )
    FMMi_n_classeAverage = np.zeros( (1,len(resultados)) )
    FMMe_n_classeAverage = np.zeros( (1,len(resultados)) )

    FMMa_n_combinacaoAverage = np.zeros( (1,len(resultados)) )
    FMMi_n_combinacaoAverage = np.zeros( (1,len(resultados)) )
    FMMe_n_combinacaoAverage = np.zeros( (1,len(resultados)) )

    One_errorAverage = np.zeros( (1,len(resultados)) )
    Coverage_errorAverage = np.zeros( (1,len(resultados)) )
    Ranking_lossAverage = np.zeros( (1,len(resultados)) )
    Average_precisionAverage = np.zeros( (1,len(resultados)) )

    for i in range(0,len(resultados)):
        erro_n_classeAverage[0,i] = resultados[i]['Erro_n_classe']
        erro_n_classe_inferiorAverage[0,i] = resultados[i]['Erro_n_classe_inferior']
        erro_n_classe_superiorAverage[0,i] = resultados[i]['Erro_n_classe_superior']

        FMMa_n_classeAverage[0,i] = resultados[i]['FMMa_n_classe']
        FMMi_n_classeAverage[0,i] = resultados[i]['FMMi_n_classe']
        FMMe_n_classeAverage[0,i] = resultados[i]['FMMe_n_classe']

        FMMa_n_combinacaoAverage[0,i] = resultados[i]['FMMa_n_combinacao']
        FMMi_n_combinacaoAverage[0,i] = resultados[i]['FMMi_n_combinacao']
        FMMe_n_combinacaoAverage[0,i] = resultados[i]['FMMe_n_combinacao']

        One_errorAverage[0,i] = resultados[i]['One_error']
        Coverage_errorAverage[0,i] = resultados[i]['CE']
        Ranking_lossAverage[0,i] = resultados[i]['RL']
        Average_precisionAverage[0,i] = resultados[i]['Average_precision']

    os.makedirs(os.path.dirname(pathResults), exist_ok = True)
    if not os.path.isfile(pathResults):#se o arquivo não existe, adiciona os labels das colunas do .csv
        fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos
        fileWrite.write('base de dados,metodo,Erro_n_classe,Erro_n_classe_inferior,Erro_n_classe_superior,FMMa_n_classe,FMMi_n_classe,FMMe_n_classe,FMMa_n_combinacao,FMMi_n_combinacao,FMMe_n_combinacao,One_error,Coverage_error,Ranking_loss,Average_precision')    
        fileWrite.close();
            
    fileWrite  = open(pathResults,"a") #abre arquivo em modo de ediçãos
        
    fileWrite.write('\n%-30s,%-30s,%1.3f,' %(datasetName, methodName, np.mean(erro_n_classeAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(erro_n_classe_inferiorAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(erro_n_classe_superiorAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(FMMa_n_classeAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(FMMi_n_classeAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(FMMe_n_classeAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(FMMa_n_combinacaoAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(FMMi_n_combinacaoAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(FMMe_n_combinacaoAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(One_errorAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(Coverage_errorAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(Ranking_lossAverage[0,:]) ))
    fileWrite.write('%1.3f,' %( np.mean(Average_precisionAverage[0,:]) ))
        
    fileWrite.close();



def wordCloudClass(dataset, target, classes, splits, language, pathResults, perc_train):

    word_very_frequently = ['pct', 'mc', 'reut', 'lisbo', 'diss', 'limited']

    # Tokens geral Dataset
    fig = plt.figure( figsize=(20,10))
    plt.axis("off")

    for i, classe in enumerate(classes):
        dt = '  '.join(dataset[target[:,i] == 1])

        for term in word_very_frequently:
            term = ' '+term+' '
            while (term in dt):
                dt = dt.replace(term, ' ')

        ax = fig.add_subplot(2,2,i+1)
        wordcloud = WordCloud(width=2000, height=1000, collocations=False, stopwords=nltk.corpus.stopwords.words(language))
        ax.imshow(wordcloud.generate(dt), interpolation='bilinear')
        ax.set_title(('%i words' % len(wordcloud.process_text(dt))), fontsize=20)
        ax.axis("off")
       
    plt.savefig(pathResults+'_1_ConjDados.eps', format='eps', dpi=1000)
    plt.savefig(pathResults+'_1_ConjDados_eps'+'.pdf')
    plt.close()

    
    
    # Tokens Dataset Train
    for test_index, train_index in splits:
        break

    fig = plt.figure( figsize=(20,10))
    plt.axis("off")
    for i, classe in enumerate(classes):
        dt = dataset[train_index]
        tg = target[train_index]

        dt = '  '.join(dt[tg[:,i] == 1])

        for term in word_very_frequently:
            term = ' '+term+' '
            while (term in dt):
                dt = dt.replace(term, ' ')

        ax = fig.add_subplot(2,2,i+1)
        wordcloud = WordCloud(width=2000, height=1000, collocations=False, stopwords=nltk.corpus.stopwords.words('portuguese'))
        ax.imshow(wordcloud.generate(dt), interpolation='bilinear')
        ax.set_title(('%i words em %i %% de treinamento' % (len(wordcloud.process_text(dt)), perc_train)), fontsize=20)
        ax.axis("off")
        
    plt.savefig(pathResults+'_2_ConjTreino.eps', format='eps', dpi=1000)
    plt.savefig(pathResults+'_2_ConjTreino_eps'+'.pdf')
    plt.close()

                    



    # Tokens Dataset Test
    tokenized_text = [word_tokenize(row) for row in dataset[train_index]] # tokenized docs
    tokens = set([item for sublist in tokenized_text for item in sublist])
    
    fig = plt.figure( figsize=(20,10))
    plt.axis("off")
    for i, classe in enumerate(classes):
        dt = dataset[test_index]
        tg = target[test_index]
    
        dt = '  '.join(dt[tg[:,i] == 1])
        
        for term in word_very_frequently:
            term = ' '+term+' '
            while (term in dt):
                dt = dt.replace(term, ' ')

        for word in tokens:
            if len(word) > 0:
                word = ' '+word+' '
                while (word in dt):
                    dt = dt.replace(word, ' ')

        ax = fig.add_subplot(2,2,i+1)
    
        wordcloud = WordCloud(width=2000, height=1000, collocations=False, stopwords=nltk.corpus.stopwords.words('portuguese'))
        ax.imshow(wordcloud.generate(dt), interpolation='bilinear')
        ax.set_title(('+%i words'  %len(wordcloud.process_text(dt))), fontsize=20)
        ax.axis("off")

    plt.savefig(pathResults+'_3_ConjTreinoIncremental.eps', format='eps', dpi=1000) #'_feedback'+feedback+}
    plt.savefig(pathResults+'_3_ConjTreinoIncremental_eps'+'.pdf') #'_feedback'+feedback+}
    plt.close()


def curvaFreqTermos(dataset, splits, pathResults, perc_train):


    # Tokens geral Dataset
    dictFreq1 = dict( Counter( ('  '.join(dataset)).split(' ') ) )
    dictposition = defaultdict()
    dictposition.default_factory = dictposition.__len__
    for key, value in sorted(dictFreq1.items(), key=lambda x:x[1], reverse = True):   
        dictposition[key]

    y = sorted(list(dictFreq1.values()), reverse = True)
    x = list(dictposition.values())
    plt.plot(x,y, 'o', label='Termos do Conj. de Dados')
    plt.xlabel("Termo/Palavra")
    plt.ylabel("Frequência")
    plt.title("Termos do treinamento inicial de %i %% e do treinamento incremental" %perc_train)
    
                    

    for test_index, train_index in splits:
        break
    
    # Tokens Dataset Train
    dictFreq2 = dict( Counter( ('  '.join(dataset[train_index])).split(' ') ) )

    y_train = []
    x_train = []
    for term in dictFreq2.keys():

        if (dictFreq1.get(term) is not None and
            dictposition[term] is not None):
            y_train.append(dictFreq1.get(term))    # Frequência
            x_train.append(dictposition[term]) # Numero do Termo

    plt.plot(x_train, y_train, '.', color='black', label='Termos do Treinamento Inicial') # , '.'



    # Tokens Dataset Test
    dictFreq3 = dict( Counter( ('  '.join(dataset[test_index])).split(' ') ) )

    y_train = []
    x_train = []
    for term in dictFreq3.keys():

        if (dictFreq2.get(term) is None and
            dictFreq1.get(term) is not None and 
            dictposition[term] is not None):

            y_train.append(dictFreq1.get(term))
            x_train.append(dictposition[term])

    plt.plot(x_train, y_train, 'r:', label='Termos adicionados com o Treinamento Incremental')
    plt.legend()  


    plt.savefig(pathResults+'_4_frequencyGraphic.eps', format='eps', dpi=1000) #'_feedback'+feedback+}
    plt.savefig(pathResults+'_4_frequencyGraphic_eps'+'.pdf') #'_feedback'+feedback+}

    plt.close()


#============================================
#Função para imprimir as médias de tempo dos folds
#============================================
def imprimiMedias_tempo(timeTrain, timeTest, classes, pathResults = None, methodName = None, datasetName = None, printResults = False):

    if timeTrain is not None and timeTest is not None:

        # Tempos
        EstatisticaClasse = np.zeros( (1,len(timeTrain)) )
        EstatisticaComb = np.zeros( (1,len(timeTrain)) )
        TreinoMetaClassificador = np.zeros( (1,len(timeTrain)) )
        RelevanciaTermos = np.zeros( (1,len(timeTrain)) )
        CurvaGaussiana = np.zeros( (1,len(timeTrain)) )
        PosicaoConfiancaRanking = np.zeros( (1,len(timeTrain)) )
        Treino = np.zeros( (1,len(timeTrain)) )

        Similaridade = np.zeros( (1,len(timeTrain)) )
        K = np.zeros( (1,len(timeTrain)) )
        Prob = np.zeros( (1,len(timeTrain)) )
        L = np.zeros( (1,len(timeTrain)) )
        Dependencia = np.zeros( (1,len(timeTrain)) )
        Teste = np.zeros( (1,len(timeTrain)) )

        for i in range(0,len(timeTrain)):

            EstatisticaClasse[0,i] = timeTrain[i]['EstatisticaClasse']
            EstatisticaComb[0,i] = timeTrain[i]['EstatisticaComb']
            TreinoMetaClassificador[0,i] = timeTrain[i]['TreinoMetaClassificador']
            RelevanciaTermos[0,i] = timeTrain[i]['RelevanciaTermos']
            CurvaGaussiana[0,i] = timeTrain[i]['CurvaGaussiana']
            PosicaoConfiancaRanking[0,i] = timeTrain[i]['PosicaoConfiancaRanking']
            Treino[0,i] = timeTrain[i]['Treino']

            Similaridade[0,i] = timeTest[i]['Similaridade']
            K[0,i] = timeTest[i]['K']
            Prob[0,i] = timeTest[i]['Prob']
            L[0,i] = timeTest[i]['L']
            Dependencia[0,i] = timeTest[i]['Dependencia']
            Teste[0,i] = timeTest[i]['Teste']


        if pathResults is not None:

            os.makedirs(os.path.dirname(pathResults.replace( '.csv', '_time.csv')), exist_ok = True)
            if not os.path.isfile(pathResults.replace( '.csv', '_time.csv')):#se o arquivo não existe, adiciona os labels das colunas do .csv
                fileWrite  = open(pathResults.replace( '.csv', '_time.csv'),"a") #abre arquivo em modo de ediçãos
                fileWrite.write('base_dados,metodo,EstatisticaClasse, EstatisticaComb,TreinoMetaClassificador,RelevanciaTermos,CurvaGaussiana,PosicaoConfiancaRanking,Treino,Similaridade,K,Prob,L,Dependencia,Teste')
                fileWrite.close();

            fileWrite  = open(pathResults.replace( '.csv', '_time.csv'),"a") #abre arquivo em modo de ediçãos

            fileWrite.write('\n%-30s,%-30s,%1.3f,' %(datasetName, methodName, np.mean(EstatisticaClasse[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(EstatisticaClasse[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(TreinoMetaClassificador[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(RelevanciaTermos[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(CurvaGaussiana[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(PosicaoConfiancaRanking[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(Treino[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(Similaridade[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(K[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(Prob[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(L[0,:]) ))
            fileWrite.write('%1.3f,' %( np.mean(Dependencia[0,:])  ))
            fileWrite.write('%1.3f,' %( np.mean(Teste[0,:])  ))

            fileWrite.close();


def imprimiResultados_Tempo(timeTrain, timeTest, metodo, nomeDataset, pathResults = None):

    if timeTrain is not None and timeTest is not None:

        nfolds = len(timeTrain)

        os.makedirs(os.path.dirname(pathResults.replace( '.csv', '_time.csv')), exist_ok = True)
        if not os.path.isfile(pathResults.replace( '.csv', '_time.csv')):#se o arquivo não existe, adiciona os labels das colunas do .csv
            fileWrite  = open(pathResults.replace( '.csv', '_time.csv'),"a") #abre arquivo em modo de ediçãos
            fileWrite.write('base_dados,metodo,EstatisticaClasse, EstatisticaComb,TreinoMetaClassificador,RelevanciaTermos,CurvaGaussiana,PosicaoConfiancaRanking,Treino,Similaridade,K,Prob,L,Dependencia,Teste')
            fileWrite.close();
            
        fileWrite  = open(pathResults.replace( '.csv', '_time.csv'),"a") #abre arquivo em modo de ediçãos
        
        for i in range(0,len(timeTrain)):          
            
            fileWrite.write('\n%-30s,%-30s,%1.3f,' %(nomeDataset, metodo, timeTrain[i]['EstatisticaClasse'] ))
            fileWrite.write('%1.3f,' %( timeTrain[i]['EstatisticaClasse'] ))
            fileWrite.write('%1.3f,' %( timeTrain[i]['TreinoMetaClassificador'] ))
            fileWrite.write('%1.3f,' %( timeTrain[i]['RelevanciaTermos'] ))
            fileWrite.write('%1.3f,' %( timeTrain[i]['CurvaGaussiana'] ))
            fileWrite.write('%1.3f,' %( timeTrain[i]['PosicaoConfiancaRanking'] ))
            fileWrite.write('%1.3f,' %( timeTrain[i]['Treino'] ))
            fileWrite.write('%1.3f,' %( timeTest[i]['Similaridade'] ))
            fileWrite.write('%1.3f,' %( timeTest[i]['K'] ))
            fileWrite.write('%1.3f,' %( timeTest[i]['Prob'] ))
            fileWrite.write('%1.3f,' %( timeTest[i]['L'] ))
            fileWrite.write('%1.3f,' %( timeTest[i]['Dependencia']  ))
            fileWrite.write('%1.3f' %( timeTest[i]['Teste']  ))

        fileWrite.close();

