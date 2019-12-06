# -*- coding: utf-8 -*- 

import numpy as np
import math

#==================================================
# Fator de confidência
#==================================================
def fatorConfidencia(frequencia, corr=None):
    #if corr is not None:
    #    frequencia = np.ceil(frequencia*corr)

    #default: k1=0.25, k2=10, k3=8
    k1=0.25 #0.25
    k2=10
    k3=8

    if len( np.shape(frequencia) )==3:
        frequencia = frequencia[0,:] # a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões

    nClasses = len(frequencia[0,:])
    nTokens = len(frequencia[:,0])

    cf = np.zeros( (nTokens,nClasses-1) )
    argmax = np.argmax(frequencia, axis=1)
   
    # Configura os elementos que serão excluidos e mantidos
    mask = ( np.zeros((nTokens,nClasses), dtype=bool) ) 
    mask[range(nTokens), argmax]  = True

    v = frequencia[mask]
    hmax = v.reshape(nTokens, 1)

    v = frequencia[ ~(mask) ]
    hmin = v.reshape(nTokens, nClasses-1)


    for it in range( 0,nTokens):
        soma = np.add.outer(hmin[it], hmax[it]).T
        sub  = np.subtract.outer(hmin[it], hmax[it]).T
        mult = np.multiply.outer(hmin[it], hmax[it]).T

        denominador = 1+(k3/soma)#(1/soma)+k3#
        cf[it] = np.where(soma == 0, 0,  ( ( ( (sub)**2 + (mult) -  (k1/(soma)) ) / (soma**2) )**k2) / denominador )
        
    cf_final = np.mean(cf,axis=1)
    return cf_final

	
#==================================================
# Fator de confidência
#==================================================
def calcula_dfs(frequencia,nTrain):

    if len( np.shape(frequencia) )==3:
        frequencia = frequencia[0,:] # a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões

    nClasses = len(frequencia[0,:])
    nTokens = len(frequencia[:,0])

    frequenciaNaoTokens = np.zeros( (nTokens,nClasses) )
    probCond_naoToken_classe = np.zeros( (nTokens,nClasses) )
    probCond_token_naoClasse = np.zeros( (nTokens,nClasses) )
    probCond_classe_token = np.zeros( (nTokens,nClasses) )
    for i in range(0,nClasses):
        
        frequenciaNaoTokens[:,i] =  nTrain[i]-frequencia[:,i]
        
        #probabilidades de não ter ocorrido o token dada a classe
        probCond_naoToken_classe[:,i]= frequenciaNaoTokens[:,i]/nTrain[i]
        
        total_ocorrencias2 = sum( nTrain ) - nTrain[i]#Quantidade de documentos em que as outras classes, tirando a atual, apareceram  
        probCond_token_naoClasse[:,i]= ( np.sum(frequencia,axis=1) - frequencia[:,i] )/total_ocorrencias2 #frequencia das outras classes tirando a atual

        #probabilidade da classe dado o token
        total_ocorrencias3 = np.sum(frequencia,axis=1) #quantidade de documentos em que cada token apareceu
        probCond_classe_token[:,i]=frequencia[:,i]/total_ocorrencias3;
        
        id2 = np.where(total_ocorrencias3==0) #se o total de ocorrencias é zero, significa que o token não apareceu em nenhuma documento
        probCond_classe_token[id2,i]=0;
    
    auxDfs = probCond_classe_token/( probCond_naoToken_classe+probCond_token_naoClasse+1 );
    #print('auxDFS',auxDfs) 
    dfs = np.sum(auxDfs,axis=1)
    #print('DFS',auxDfs) 
    
    id2 = np.where(dfs==0) #encontra os id onde dfs=0
    dfs[id2] = 0.5;
    
    #normaliza dfs entre 0 e 1
    dfs = (dfs-0.5)/(1-0.5)
          
    return np.transpose(dfs)

            

            
#==================================================
# Information Gain
#==================================================
def calcula_informationGain3(frequencia,frequencianC, nTrain, nTrainTotal):

    if len( np.shape(frequencia) )==3:
        frequencia = frequencia[0,:]# a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões
	
    nClasses = len(frequencia[0,:])
    nTokens = len(frequencia[:,0])

    frequenciaNaoTokens = np.zeros( (nTokens,nClasses) )
    probCond_token_classe = np.zeros( (nTokens,nClasses) )
    probCond_naoToken_classe = np.zeros( (nTokens,nClasses) )
    prob_classe = np.zeros( nClasses )
    
    for i in range(nClasses):
        #for j in range(nClasses):
        #    if j!=i:
        #        frequenciaNaoTokens[:,i] +=  frequencia[:,j]
        #        probCond_naoToken_classe[:,i] += frequencia[:,j]/nTrain[j]

        #probabilidades do token dada a classe
        probCond_token_classe[:,i] = frequencia[:,i]/nTrain[i];

        #probabilidades de não ter ocorrido o token dada a classe
        probCond_naoToken_classe[:,i] = frequencianC[:,i]/(nTrainTotal-nTrain[i])

        #probabilidade da classe
        prob_classe[i] = nTrain[i]/sum(nTrain[:]);

    prob_token = np.sum(frequencia,axis=1) / nTrainTotal#sum(nTrain[:]);

    prob_naoToken = np.sum(frequenciaNaoTokens,axis=1) / sum(nTrain[:]);

    auxGain3 = np.zeros( (nTokens,nClasses) )
    for i in range(nClasses):
        probIntersection_token_classe = probCond_token_classe[:,i]  *prob_classe[i];
        probIntersection_naoToken_classe = probCond_naoToken_classe[:,i]  *prob_classe[i];

        auxGain1 = probIntersection_token_classe * np.log2( probIntersection_token_classe/(prob_token*prob_classe[i]) );
        auxGain2 = probIntersection_naoToken_classe*np.log2( probIntersection_naoToken_classe/(prob_naoToken*prob_classe[i]) );

        auxGain1[ np.isnan(auxGain1) ] = 0;#onde tive NaN coloca 0
        auxGain2[ np.isnan(auxGain2) ] = 0;#onde tive NaN coloca 0
        
        auxGain3[:,i] = auxGain1+auxGain2;


    gain = np.sum( auxGain3,axis=1 );

    return gain
            


#==================================================
# Information Gain
#==================================================
def calcula_informationGain2(frequencia,nTrain):

    if len( np.shape(frequencia) )==3:
        frequencia = frequencia[0,:]# a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões
	
    nClasses = len(frequencia[0,:])
    nTokens = len(frequencia[:,0])

    frequenciaNaoTokens = np.zeros( (nTokens,nClasses) )
    probCond_token_classe = np.zeros( (nTokens,nClasses) )
    probCond_naoToken_classe = np.zeros( (nTokens,nClasses) )
    prob_classe = np.zeros( nClasses )
    
    for i in range(nClasses):
        frequenciaNaoTokens[:,i] =  nTrain[i]-frequencia[:,i]

        #probabilidades do token dada a classe
        probCond_token_classe[:,i] = frequencia[:,i]/nTrain[i];

        #probabilidades de não ter ocorrido o token dada a classe
        probCond_naoToken_classe[:,i]= frequenciaNaoTokens[:,i]/nTrain[0,i]

        #probabilidade da classe
        prob_classe[i] = nTrain[i]/sum(nTrain);

    prob_token = np.sum(frequencia,axis=1) / sum(nTrain);

    prob_naoToken = np.sum(frequenciaNaoTokens,axis=1) / sum(nTrain);

    auxGain3 = np.zeros( (nTokens,nClasses) )
    for i in range(nClasses):
        probIntersection_token_classe = probCond_token_classe[:,i]*prob_classe[i];
        probIntersection_naoToken_classe = probCond_naoToken_classe[:,i]*prob_classe[i];

        auxGain1 = probIntersection_token_classe * np.log2( probIntersection_token_classe/(prob_token*prob_classe[i]) );
        auxGain2 = probIntersection_naoToken_classe*np.log2( probIntersection_naoToken_classe/(prob_naoToken*prob_classe[i]) );

        auxGain1[ np.isnan(auxGain1) ] = 0;#onde tive NaN coloca 0
        auxGain2[ np.isnan(auxGain2) ] = 0;#onde tive NaN coloca 0
        
        auxGain3[:,i] = auxGain1+auxGain2;


    gain = np.sum( auxGain3,axis=1 );

    return gain
    


#==================================================
# OR
#==================================================
def OR(frequencia, frequencianC, nTrain, cj, nTrainTotal):

    if len( np.shape(frequencia) )==3:
        frequencia = frequencia[0,:]# a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões
	
    nClasses = len(frequencia[0,:])
    nTokens = len(frequencia[:,0])

    frequenciaNaoTokens = np.zeros( (nTokens,nClasses) )
    probCond_token_classe = np.zeros( (nTokens,nClasses) )
    probCond_naoToken_classe = np.zeros( (nTokens,nClasses) )
    probCond_token_Nclasse = np.zeros( (nTokens,nClasses) )
    prob_classe = np.zeros( nClasses )
    
    #for i in range(nClasses):
    frequenciaNaoTokens[:,cj] =  nTrain[cj]-frequencia[:,cj]

    #probabilidades do token dada a classe
    probCond_token_classe[:,cj] = frequencia[:,cj]/nTrain[cj];


    #probabilidades do token dada a classe
    probCond_token_Nclasse[:,cj] = frequencianC[:,cj]/(nTrainTotal-nTrain[cj]);



    #probabilidades de não ter ocorrido o token dada a classe
    probCond_naoToken_classe[:,cj]= frequenciaNaoTokens[:,cj]/ nTrain[cj]

    gain = ( probCond_token_classe[:,cj] * (1 - probCond_token_Nclasse[:,cj]) ) / ( (1-probCond_token_classe[:,cj]) * (probCond_token_Nclasse[:,cj]) )

    return gain



#==================================================
# DFG
#==================================================
def calcula_dfg(freqTokens, nTrainTotal):
    
    prob_token = freqTokens / freqTokens;

    return prob_token

                
            

#==================================================
# Information Gain
#==================================================
def calcula_informationGainML(frequencia_a, frequencia_b, nTrain, trainTotal): # BR tem média

    if len( np.shape(frequencia_a) )==3:
        frequencia_a = frequencia_a[0,:]# a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões
	
    nClasses = len(frequencia_a[0,:])
    nTokens = len(frequencia_a[:,0])

    frequencia_c = np.zeros( (nTokens,nClasses) )
    frequencia_d = np.zeros( (nTokens,nClasses) )

    frequencia_c =  nTrain-frequencia_a
    frequencia_d =  (trainTotal-nTrain)-frequencia_b

    N = frequencia_a + frequencia_b + frequencia_c + frequencia_d

    p1 = (frequencia_a/N) * np.log2( (frequencia_a*N) / ((frequencia_a+frequencia_b)*(frequencia_a + frequencia_c)) ) 
    p2 = (frequencia_c/N) * np.log2( (frequencia_c*N) / ((frequencia_c+frequencia_d)*(frequencia_a + frequencia_c)) ) 
    p3 = (frequencia_b/N) * np.log2( (frequencia_b*N) / ((frequencia_a+frequencia_b)*(frequencia_b + frequencia_d)) ) 
    p4 = (frequencia_d/N) * np.log2( (frequencia_d*N) / ((frequencia_c+frequencia_d)*(frequencia_b + frequencia_d)) )

    p1[ np.isnan(p1) ] = 0
    p2[ np.isnan(p2) ] = 0
    p3[ np.isnan(p3) ] = 0
    p4[ np.isnan(p4) ] = 0

    gain = p1 + p2 + p3 + p4

    gain_final = normalization(gain.sum(axis=1))#np.mean(gain,axis=1)
    return gain_final




#==================================================
# Fator de confidência
#==================================================
def fatorConfidenciaBR(frequencia, frequencianC):
    #default: k1=0.25, k2=10, k3=8
    k1=0.25 #0.25
    k2=10
    k3=8

    if len( np.shape(frequencia) )==3:
        frequencia = frequencia[0,:] # a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões

    nClasses = len(frequencia[0,:])
    nTokens = len(frequencia[:,0])

    cf = np.zeros( (nTokens,nClasses) )
    for it in range( 0,nTokens):

        for i in range( 0,nClasses ):   

            hmax = frequencia[it,i]
            hmin = frequencianC[it,i]

            if ( (hmax!=0) or (hmin!=0) ):
                difh=hmax-hmin
                sh=hmax+hmin
                cf[it,i] = ( ( (difh)**2 + (hmax*hmin) - (k1/sh))/(sh**2) )**k2
                denominador = 1+(k3/sh) #(1+k3*sh)/sh 

                cf[it,i]=cf[it,i]/denominador

            else:#se hmax for igual a 0, significa que o token não aparece para nenhuma das classes, logo seu cf é 0
                cf[it,i]=0


    cf_final = np.mean(cf,axis=1)
    return cf_final
	


#==================================================
# chi-squared
#==================================================
def calcula_x2(frequencia_a, frequencia_b, nTrain, trainTotal):

    if len( np.shape(frequencia_a) )==3:
        frequencia_a = frequencia_a[0,:]# a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões
	
    nClasses = len(frequencia_a[0,:])
    nTokens = len(frequencia_a[:,0])

    frequencia_c = np.zeros( (nTokens,nClasses) )
    frequencia_d = np.zeros( (nTokens,nClasses) )

    frequencia_c =  nTrain-frequencia_a
    frequencia_d =  (trainTotal-nTrain)-frequencia_b

    N = frequencia_a + frequencia_b + frequencia_c + frequencia_d

    gain = N * (((frequencia_a*frequencia_d) - (frequencia_b*frequencia_c))**2) / ( (frequencia_a+frequencia_b)*(frequencia_a+frequencia_c)*(frequencia_b+frequencia_d)*(frequencia_c+frequencia_d)  )
    
    gain[ np.isnan(gain) ] = 0

    gain = normalization(gain)

    gain_final = np.mean(gain,axis=1)
    return gain_final


#==================================================
# Information Gain
#==================================================
def calcula_GSS(frequencia_a, frequencia_b, nTrain, trainTotal):

    if len( np.shape(frequencia_a) )==3:
        frequencia_a = frequencia_a[0,:]# a matrix de frequencia tem 3 dimensões, então tem que transformar em 2 dimensões
	
    nClasses = len(frequencia_a[0,:])
    nTokens = len(frequencia_a[:,0])

    frequencia_c =  nTrain-frequencia_a
    frequencia_d =  (trainTotal-nTrain)-frequencia_b

    N = frequencia_a + frequencia_b + frequencia_c + frequencia_d

    gain = ((frequencia_a*frequencia_d) - (frequencia_b*frequencia_c)) / ( N**2 )
    
    gain[ np.isnan(gain) ] = 0

    gain = normalization(gain)
    return gain



def normalization(value):

    return (value - value.min(axis=0)) / (value.max(axis=0) - value.min(axis=0))
