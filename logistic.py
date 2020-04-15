import pandas as pd
import numpy as np
import os,re
cwd=os.getcwd()
import matplotlib.pyplot as plt 
import seaborn as sn
import pickle
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction as fe
from scipy.sparse import hstack

train_file = cwd+r"/snli_1.0/snli_1.0/snli_1.0_train.csv"
test_file= cwd+r"/snli_1.0/snli_1.0/snli_1.0_test.csv"
train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)
train_labels=np.array((train_data['label']))
test_labels=np.array(test_data['label'])

train_data_mat=train_data[['sentence1','sentence2']].values
test_data_mat=test_data[['sentence1','sentence2']].values

# vectorizer = fe.text.TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')
# train_corpus_tf_idf = vectorizer.fit_transform(X_train)

vectorizer1 = fe.text.TfidfVectorizer( sublinear_tf=True, use_idf =True,analyzer='word',ngram_range=(1,2))
vectorizer2 = fe.text.TfidfVectorizer( sublinear_tf=True, use_idf =True,analyzer='word',ngram_range=(1,2))

train_sent1=vectorizer1.fit_transform(train_data_mat[:,0])
train_sent2=vectorizer2.fit_transform(train_data_mat[:,1])

test_sent1=vectorizer1.transform(test_data_mat[:,0])
test_sent2=vectorizer2.transform(test_data_mat[:,1])

train_data=hstack((train_sent1,train_sent2))
test_data=hstack((test_sent1,test_sent2))

clf = LogisticRegression(random_state=0,max_iter=10000,multi_class='ovr')
filename=cwd+"/Logistic_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
score = loaded_model.score(test_data,test_labels)
print(score)
#result = loaded_model.predict(test_data)
#
#from sklearn.metrics import confusion_matrix
#totalMatNB = np.zeros((3,3))
## confusion_matrix(True,predicted)
#totalMatNB = totalMatNB + confusion_matrix(test_labels, result)
#totalNB=0
#totalNB = totalNB+sum(test_labels==result)
#plt.figure(figsize=(10, 7))
#mapping={}
#mapping['neutral']=0
#mapping['contradiction']=1
#mapping['entailment']=2
#xticklabels=['neutral','contradiction','entailment']
#fig = sn.heatmap(confusion_matrix(test_labels, result), xticklabels=xticklabels,yticklabels=xticklabels,annot=True)
#plt.title("Confusion Matrix for TFIDF")
#plt.show()


