# -*- coding: utf-8 -*- 
def getDefault(listElement, element = None, listElementExecuted = None):

    if listElementExecuted is not None:
        listElement = [x for x in listElement if x not in listElementExecuted]
    
    if element is not None:
        return [x for x in listElement if x[0] in element]
    else:
        return listElement


def getVocabularies(learningType, vocabulary = None):

    vocabularies = ['Fixo']

    return getDefault(vocabularies, vocabulary)

def getFeedbacks(learningType, feedback=None):

    feedbacks = ['Imediato',
                 'Atraso',
                 'Incerto',
                 'Incert-Atras']

    return getDefault(feedbacks, feedback)

def getPerformExperimentTypes(learningType, performExperiment=None):

    performExperimentTypes = ['CrossValidation']

    return getDefault(performExperimentTypes, performExperiment)

    
def getDefautMethods(learningType, vocabulary=None, method=None, methodsExecuted=None):

    methods = [	['M.NB_1vsAll','binary'],
				['B.NB_1vsAll','binary'],
				['PA_1vsAll','TFIDF'],
                ['SGD_1vsAll','TFIDF'],
				['Perceptron_1vsAll','TFIDF'],
                ['M.NB_TP-CC','binary'],
                ['B.NB_TP-CC','binary'],
				['PA_TP-CC','TFIDF'],
                ['SGD_TP-CC','TFIDF'],
                ['Perceptron_TP-CC','TFIDF'],
                ['MLP','TFIDF'],
				['MDLText_MD/SGD','TFIDF'],
                ['MDLText_MD_V2/SGD','TFIDF']
        ]
        
    return getDefault(methods, method, methodsExecuted)

def getDefautPathResults(learningType=None, vocabulary=None, performExperimentType=None):
    
    if learningType is None:
        directory = 'results'
    else:
        directory = 'results/'+learningType

    if vocabulary is None:
        directory = (directory)
    else:
        directory = (directory+'_vocabulary'+vocabulary)

    if performExperimentType is None:
        directory = (directory+'/')
    else:
        directory = (directory+'/'+performExperimentType+'/')
  

    return directory

def getDefautPathLatexScript():
    return ('results/')

def getDefautDatasets(dataset=None):

    datasets  = [	['reuters_stopWords_stemming_orgs','txt',0,'rorg', 'english'],
					['reuters_stopWords_stemming_places','txt',0,'rplaces', 'english'],
					['rcv1_stopWords_stemming_nivel1','txt',0,'rcv1nv1', 'english'],
					['rcv1_stopWords_stemming_nivel2','txt',0,'rcv1nv2', 'english'],
					['rcv2_portuguese_stopWords_stemming_nivel1','txt',0,'rcv2pt1', 'portuguese'],
					['rcv2_portuguese_stopWords_stemming_nivel2','txt',0,'rcv2pt2', 'portuguese'],
					['rcv2_spanish_stopWords_stemming_nivel1','txt',0,'rcv2sp1', 'spanish'],
					['rcv2_spanish_stopWords_stemming_nivel2','txt',0,'rcv2sp2', 'spanish'],
					['rcv2_italian_stopWords_stemming_nivel1','txt',0,'rcv2it1', 'italian'],
					['rcv2_italian_stopWords_stemming_nivel2','txt',0,'rcv2it2', 'italian'],
					['delicious', 'mulan', 983, 'delicious', 'english'],
					['tmc2007', 'mulan', 22, 'tmc2007', 'english'],
					['bibtex', 'mulan', 159, 'bibtex', 'english'],
					['medical', 'mulan', 45, 'medical', 'english'],
					['enron', 'mulan', 53, 'enron', 'english']
				]
				

    return getDefault(datasets, dataset)

def getDefautMeasures():

    measures = [
                ['FMMa','F-medidaMacro', 'max', True],
                ['FMMi','F-medidaMicro', 'max', True],
                ['SMa','SensitividadeMacro', 'max', True],
                ['SMi','SensitividadeMicro', 'max', True],
                ['PMa','PrecisaoMacro', 'max', True],
                ['PMi','PrecisaoMicro', 'max', True],
                ['HL','hammingLoss', 'min', True],
                ['SAcc','subset_accuracy', 'max', True],
                ['Acc','accuracy_index', 'max', True],
                ['CE','coverage_error', 'min', False],
                ['RL','rankingLoss', 'min', False],
                ['Tempo','tempo', 'min', True],
                ['Err\%','Erro_n_classe', 'min', False],
                ['Err-\%','Erro_n_classe_inferior', 'min', False],
                ['Err+\%','Erro_n_classe_superior', 'min', False],
                ['FMMaNC','FMMa_n_classe', 'max', False],
                ['FMMi_NClasse','FMMi_n_classe', 'max', False],
                ['FMMeNC','FMMe_n_classe', 'max', False],
                ['1E','One_error', 'min', False],
                ['AP','Average_precision', 'max', False],
                ['FMMe','F-medida', 'max', False],
                ['SMe','Sensitividade', 'max', False],
                ['PMe','Precisao', 'max', False]              
                ]

    return measures
