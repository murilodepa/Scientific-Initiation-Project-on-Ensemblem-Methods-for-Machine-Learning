import ISee
import Profiler

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
def multisub(patterns, strns, text_pos):
    for strn in strns:
        for pattern, subst in patterns:
            strn[text_pos] = re.sub(pattern, subst, strn[text_pos])
    return strns

#A: bigrams
#B: all characters to lowercase
#C: accentuation removal
#D: special character treatment
#E: stop-words removal
#F: twitter user names removal
#G: twitter topics removal
#H: Reduction of laugh expressions to a common token
laugh_token = '_laugh_'
heuristics = {
    'A': [('not( [^ ]+)', 'not_\1')],
    'B': [('([A-Z]*)', lambda x: x.group(0).lower())],
    'C': [('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('à', 'a'), ('ã', 'a'), ('ẽ', 'e'), ('ĩ', 'i'), ('õ', 'o'), ('ũ', 'u'),  ('â', 'a'), ('ê', 'e'), ('ñ', 'n')],
    'D': [(':\)', ''), (':\(', ''), (':\^\)', ''), (':p', ''), (':3', ''), (':c', ''), ('c:', ''), (':o', '')],
    'E': [(' the ', ' '), (' a ', ' '), (' e ', ' '), (' o ', ' ')],
    'F': [('@[^ ]+', ''), ('@[^ ]+ ', '')],
    'G': [('#[^ ]+', ''), ('#[^ ]+ ', '')],
    'H': [('[ashu]{3,}', laugh_token), ('[ah]{3,}', laugh_token), ('[eh]{3,}', laugh_token), ('k{4,}', laugh_token), ('[rs]{3,}', laugh_token), ('[ehu]{3,}', laugh_token)],
}

heuristics_in_use = '' #ABCDEFGH

max_features = 1000
embedding_dims = 50
maxlen = 50

#train_on_tweets = True

#split_tokenizer_cats = True
#split_classifier_cats = True

kfold = 5

will_shuffle = True
shuffle_seed = 1234

n_est = 1000

prf = Profiler.Milestones()

print("Creating ensemble")
ensemble = ISee.Ensemble()
prf.add_milestone("Created ensemble object")

print("Adding models")
##ensemble.add('Random', NullHypothesis.RandomClassifier())
ensemble.add('XGB',  XGBClassifier(n_estimators=n_est, learning_rate=0.01, max_depth=6, subsample=0.65, colsample_bytree=0.25, gamma=5))
ensemble.add('Forest', RandomForestClassifier(n_estimators=90))
ensemble.add('Naive-B',  GaussianNB())
#ensemble.add('Naive-B_F',  GaussianNB(), uses_one_hot=False)
ensemble.add('SVM_T', svm.SVC(gamma='scale'))
#ensemble.add('Linear_T', LinearRegression())
#ensemble.add('SVM_F', svm.SVC(gamma='scale'), uses_one_hot=False)
#ensemble.add('Linear_F', LinearRegression(), uses_one_hot=False)

ensemble.add('RNC', Code_vin.RNC_vin(max_features=max_features, embedding_dims=embedding_dims, maxlen=maxlen, filters=100, kernel_size=3, hidden_dims=250, output_dims=2), uses_one_hot=False, uses_categorical=True, uses_argmax=True)
prf.add_milestone("Added models")

print("Setting encoder & tokenizer")
ensemble.set_encoder(LabelEncoder())
ensemble.set_tokenizer(Tokenizer(max_features))
prf.add_milestone("Set encoder & tokenizer")

print("Reading file")
text_position = 0
class_position = 1
#file_name = "brexit_blog_corpus.csv"
file_name = "tweets_br.csv"
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

print(str(lines[15]))
textos  = [str(a[text_position])  for a in lines]
classes = [str(b[class_position]) for b in lines]
indexes = [c[-1] for c in lines]
prf.add_milestone("Aggregated data")

print("Setting data")
ensemble.set_datasets(textos, classes, indexes)
prf.add_milestone("Set datasets")

print("Fitting encoder & tokenizer")
ensemble.fit_encoder()
ensemble.fit_tokenizer()
prf.add_milestone("Fitted encoder & tokenizer")

#phrases = [
#'mas hoje eu to com um sentimento terrivel',
#'No ar, Cristiano Araujo - Hoje Eu To Terrivel',
#'hoje eu to terivel',
#'Pra romance é off, pra balada é disponível, hoje eu to terrível ...',
#'Hoje eu tava terrível na aula do curso',
#'Hoje eu To terrível',
#'Hoje eu tô terrível !! muito terrível',
#]

#print(str(ensemble.tokenizer.texts_to_sequences(phrases)))
#print("Encoder params = " + str(ensemble.encoder.get_params()))

print("Splitting data")
ensemble.split_data(shuffle_seed=1337, purge_duplicates=False)
prf.add_milestone("Split data")

#ensemble.test_split()
#ensemble.test_sets()
#exit(0)

print("Training ensemble")
ensemble.train(max_features, maxlen, profiler=prf)
prf.add_milestone("Trained ensemble")

print("Testing ensemble")
res = ensemble.test(max_features, maxlen, profiler=prf)
prf.add_milestone("Tested ensemble")

#print("Res = " + str(res))

ensemble.evaluate(verbose=True)
prf.add_milestone("Finished evaluating")

print("Profiler data:")
prf.exhibit(max_d=10)
