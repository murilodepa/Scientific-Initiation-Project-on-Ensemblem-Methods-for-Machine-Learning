# -*- coding: utf-8 -*-
import ISee
import Profiler
from argy import Args

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer

import numpy as np

import Code_vin
import NullHypothesis

import re #REGEX sera utilizado para implementar as heuristicas de pre-processamento
import random

#Classe utilizada para analisar os parametros passados por linha de comando
import sys
args = Args(sysargs=sys.argv)

#Configuracao da parametrizacao da linha de comando
args.add('-h', param_type=str, default_value='ABCDEFGH') # Heuristics in use
args.add('-mf', param_type=int, default_value=1000) # Max features
args.add('-ed', param_type=int, default_value=50) # Embedding dimensions
args.add('-k', param_type=int, default_value=5) # K-folding
args.add('--shuffle', param_type=str, default_value=None) # Will shuffle
args.add('-s', param_type=int, default_value=1234) # Shuffle seed
args.add('-d', param_type=str, default_value='portugues') # Dataset
args.add('-nc', param_type=int, default_value=7) # Number of classifiers

#Funcao para analisar multiplos padroes REGEX em cada string
def multisub(patterns, strns, text_pos):
    for strn in strns:
        for pattern, subst in patterns:
            strn[text_pos] = re.sub(pattern, subst, strn[text_pos])
    return strns

#Codigos do pre-processamento (baseados no projeto de IC anterior)
#A: bigrams
#B: all characters to lowercase
#C: accentuation removal
#D: special character treatment
#E: stop-words removal
#F: twitter user names removal
#G: twitter topics removal
#H: Reduction of laugh expressions to a common token
laugh_token = ' _laugh_ '
heuristics = {
    'A': [('not( [^ ]+)', 'not_\1')],
    'B': [('([A-Z]*)', lambda x: x.group(0).lower())],
    'C': [('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('à', 'a'), ('ã', 'a'), ('ẽ', 'e'), ('ĩ', 'i'), ('õ', 'o'), ('ũ', 'u'),  ('â', 'a'), ('ê', 'e'), ('ñ', 'n')],
    'D': [(':\)', ''), (':\(', ''), (':\^\)', ''), (':p', ''), (':3', ''), (':c', ''), ('c:', ''), (':o', '')],
    'E': [(' the ', ' '), (' a ', ' '), (' e ', ' '), (' o ', ' ')],
    'F': [('@[^ ]+', ''), ('@[^ ]+ ', '')],
    'G': [('#[^ ]+', ''), ('#[^ ]+ ', '')],
    'H': [(' [ashu]{3,} ', laugh_token), (' [ah]{3,} ', laugh_token), (' [eh]{3,} ', laugh_token), (' k{4,} ', laugh_token), (' [rs]{3,} ', laugh_token), (' [ehu]{3,} ', laugh_token)],
}

#Configuracoes
#heuristics_in_use = 'ABCDEFGH' #ABCDEFGH
#max_features = 1000
#embedding_dims = 50
#maxlen = 50
#kfold = 5
#will_shuffle = True
#shuffle_seed = 1234
#file_alias = 'portugues' #'portugues'
heuristics_in_use = args.get('-h')
max_features = args.get('-mf')
embedding_dims = args.get('-ed')
maxlen = embedding_dims
kfold = args.get('-k')
will_shuffle = (args.get('--shuffle') != None)
shuffle_seed = args.get('-s')
file_alias = args.get('-d')
n_models = args.get('-nc')

prf = Profiler.Milestones()

#Criando o ensemble
ensemble = ISee.Ensemble(n_folds=kfold)
prf.add_milestone("Created ensemble object")

#Criando encoder (para as classes) e tokenizador (para os textos)
ensemble.set_encoder(LabelEncoder())
ensemble.set_tokenizer(Tokenizer(max_features))
prf.add_milestone("Set encoder & tokenizer")

if file_alias == 'brexit':
    text_position = 4
    class_position = 5
    file_name = "brexit_blog_corpus.csv"
elif file_alias == 'portugues':
    text_position = 2
    class_position = 1
    file_name = "kaggle.csv"
else:
    print("File alias nao reconhecido (-d)")
    exit()
    
#Carregando os dados do arquivo de dataset
with open("datasets/" + file_name) as f:
    lines = [line.rstrip('\n').split('\t') for line in f][1:]
    print("Read {} lines".format(len(lines)))
    print("Applying heuristics " + heuristics_in_use + " (" + str(len(heuristics_in_use)) + ")")
    for h in heuristics_in_use:
        lines = multisub(heuristics[h], lines, text_position)
    line_no = 0
    for line in lines:
        line.append(line_no)
        line_no += 1
prf.add_milestone("Read file")

#Embaralhando o dataset (se houver o parametro --shuffle)
if will_shuffle:
    print("Shuffling lines with seed {}".format(shuffle_seed))
    random.seed(shuffle_seed)
    random.shuffle(lines)

#Separando textos e classes
textos  = [str(a[text_position])  for a in lines]
classes = [str(b[class_position]) for b in lines]
n_classes = len(set(classes))
indexes = [c[-1] for c in lines]
prf.add_milestone("Aggregated data")

#Configurando os datasets dentro do ensemble
ensemble.set_datasets(textos, classes, indexes)
prf.add_milestone("Set datasets")

#Configurando encoder e tokenizador
ensemble.fit_encoder()
ensemble.fit_tokenizer()
prf.add_milestone("Fitted encoder & tokenizer")

#Separando os dados para k-folding
ensemble.split_data(purge_duplicates=False)
prf.add_milestone("Split data")

#Adicionando os modelos de classificadores
for i in range(n_models):
    ensemble.add('RNC_{}'.format(i + 1), Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=n_classes), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
#ensemble.add('RNC_binary_crossentropy_1', Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=n_classes, compile_loss='binary_crossentropy'), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
#ensemble.add('RNC_binary_crossentropy_2', Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=n_classes, compile_loss='binary_crossentropy'), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
#ensemble.add('RNC_binary_crossentropy_3', Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=n_classes, compile_loss='binary_crossentropy'), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
#ensemble.add('RNC_categorical_crossentropy_1', Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=n_classes, compile_loss='categorical_crossentropy'), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
#ensemble.add('RNC_categorical_crossentropy_2', Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=n_classes, compile_loss='categorical_crossentropy'), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
#ensemble.add('RNC_categorical_crossentropy_3', Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=n_classes, compile_loss='categorical_crossentropy'), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
prf.add_milestone("Added models")

#Treinando os classificadores
ensemble.train(max_features, maxlen, profiler=prf)
prf.add_milestone("Trained ensemble")

#Guardando resultados da avaliacao
#with open('eval_results_' + file_name[:-4] + '.txt', 'a') as f:
#    f.write(ensemble.evalHistories())

#Testando os classificadores com os textos nao usados (kfolding)
res = ensemble.test(max_features, maxlen, profiler=prf)
prf.add_milestone("Tested ensemble")

#Avaliando os resultados dos testes
ensemble.evaluate(verbose=False, file_d=open('manual_results_' + file_name[:-4] + '.txt', 'a'), csv_file=open('results_' + file_name[:-4] + '.csv', 'w'))
prf.add_milestone("Finished evaluating")

#Exibindo resultados de tempo de execucao
print("Profiler data:")
prf.exhibit(max_d=10)

