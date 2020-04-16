import getopt, sys ,os
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor,optim
from torch.autograd import Variable
import pickle
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from torchsummary import summary
from scipy.sparse import hstack
# from torch_utils import AverageMeter
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction as fe
import pickle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from torch.utils.data import TensorDataset, DataLoader
import re
maxlen=50
vocab_size=31292
device=torch.device('cpu')

def load_checkpoint(filepath):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    model_path=os.path.join(__location__,filepath)
    if os.path.isfile(model_path):
        print("Model Already trained and File exists")
        checkpoint = torch.load(filepath)
        
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
    else:
        return Net()    
from torch.autograd import Variable
class LSTM_2(nn.Module):
  def __init__(self, options,pretrained_model): 
    super(LSTM_2, self).__init__()
    weights = torch.FloatTensor(pretrained_model.wv.vectors)
    self.embedding = nn.Embedding(options['vocab_size'], options['embed_dim']).from_pretrained(weights)
    self.projection = nn.Linear(options['embed_dim'], 300)
    self.dropout = nn.Dropout(p = options['dp_ratio'])
    self.lstm = nn.LSTM(300, options['d_hidden'], 3,bidirectional=True)
    self.relu = nn.ReLU()
    self.out = nn.Sequential(
      nn.Linear(2048, 1024),
      self.relu,
      self.dropout,
      nn.Linear(1024, 1024),
      self.relu,
      self.dropout,
      nn.Linear(1024, 1024),
      self.relu,
      self.dropout,
      nn.Linear(1024, options['out_dim'])
    )
    # pass
  def forward(self, premise,hypothesis):
    # print(premise)
    # print(premise.shape)
    global maxlen
    global vocab_size
    global device
    self.hidden=self.init_hidden()
    seq_len_prem=torch.zeros(premise.shape[0]).to(device)
    seq_len_hypo=torch.zeros(premise.shape[0]).to(device)
    # print(premise_embed)
    try:
      for i in range(premise.shape[0]):
        if premise[i][-1]!=vocab_size-1 or premise[i][0]==vocab_size-1:
          seq_len_prem[i]=maxlen
          continue
        # print(torch.where(batch[i]==vocab_size-1))
        seq_len_prem[i]=torch.where(premise[i]==vocab_size-1)[0][0].tolist()
        if seq_len_prem[i]==0:
          print(premise[i])
    except:
      print(premise[i])
      pass
    try:
      for i in range(premise.shape[0]):
        if hypothesis[i][-1]!=vocab_size-1 or hypothesis[i][0]==vocab_size-1:
          seq_len_hypo[i]=maxlen
          continue
        # print(torch.where(batch[i]==vocab_size-1))
        seq_len_hypo[i]=torch.where(hypothesis[i]==vocab_size-1)[0][0].tolist()
        if seq_len_hypo[i]==0:
          print(hypothesis[i])
    except:
      print(hypothesis[i])
      pass
    # print(seq_len_prem)
    premise_embed = self.embedding(premise)
    hypothesis_embed = self.embedding(hypothesis)
    premise_proj = self.relu(self.projection(premise_embed))
    hypothesis_proj = self.relu(self.projection(hypothesis_embed))
    try:
      premise_proj= torch.nn.utils.rnn.pack_padded_sequence(premise_proj, seq_len_prem, batch_first=True,enforce_sorted=False)
      hypothesis_proj= torch.nn.utils.rnn.pack_padded_sequence(hypothesis_proj, seq_len_hypo, batch_first=True,enforce_sorted=False)
    except:
      print(seq_len_hypo)
      print(seq_len_prem)
    encoded_premise, _ = self.lstm(premise_proj,self.hidden)
    encoded_hypothesis, _ = self.lstm(hypothesis_proj,self.hidden)
    try:
      encoded_premise, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded_premise, batch_first=True,total_length=maxlen)
      encoded_hypothesis, _ = torch.nn.utils.rnn.pad_packed_sequence(encoded_hypothesis, batch_first=True,total_length=maxlen)
    except:
      print(encoded_premise)
      print(encoded_hypothesis)
      # pass
    premise = encoded_premise.sum(dim = 1)
    hypothesis = encoded_hypothesis.sum(dim = 1)
    combined = torch.cat((premise, hypothesis), 1)
    return self.out(combined)

  def init_hidden(self):
      # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
      hidden_a = torch.randn(2*2, 128,50)
      hidden_b = torch.randn(2*2, 128,50)
      
      # hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        
   
      hidden_a = Variable(hidden_a)
      hidden_b = Variable(hidden_b)
  def LSTM_2(options):
    return LSTM_2(options)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def test_deep(model, testloader):
    """ Training the model using the given dataloader for 1 epoch.
    Input: Model, Dataset, optimizer,
    """
    mapping2={}
    mapping2[0]='neutral'
    mapping2[1]='contradiction'
    mapping2[2]='entailment'
    n_correct, n_total, n_loss = 0, 0, 0
    confusion_matrix = torch.zeros(3, 3)
    model.eval()
    avg_loss = AverageMeter("average-loss")
    y_gt = []
    y_pred_label = []
    # hidden=model.init_hidden()
    criterion = nn.CrossEntropyLoss()
    global device
    batch_size=128
    with torch.no_grad():
      for batch_idx, (premise,hypothesis,target) in enumerate(testloader):
        premise=premise.to(device).to(torch.int64)
        hypothesis=hypothesis.to(device).to(torch.int64)
        target=target.to(device).to(torch.int64)
        # _,target=target.max(dim=1)
        answer = model(premise,hypothesis)
        loss = criterion(answer, target.long())

        pred = torch.max(answer, 1)[1].view(target.size())
        for t, p in zip(target, pred):
          confusion_matrix[t.long()][p.long()] += 1
        
        n_correct += (pred == target).sum().item()
        n_total += batch_size
        n_loss += loss.item()
        y_gt+=list(mapping2[int(i)] for i in target.numpy())
        y_pred_label+=list(mapping2[int(i)] for i in pred.numpy())
      test_loss = n_loss/n_total
      test_acc = 100. * n_correct/n_total

    return test_loss, y_gt, y_pred_label

def gen_graph(gt,pred):
    op = confusion_matrix(gt, pred)
    print(op)
    plt.figure(figsize=(10, 7))
    fig = sn.heatmap(op, annot=True)
    plt.title("Confusion Matrix for MLP")
    plt.show()
def test_logistic(model,train_data,test_data,test_labels):
    train_data_mat=train_data[['sentence1','sentence2']].values
    test_data_mat=test_data[['sentence1','sentence2']].values
    vectorizer1 = fe.text.TfidfVectorizer( sublinear_tf=True, use_idf =True,analyzer='word',ngram_range=(1,2))
    vectorizer2 = fe.text.TfidfVectorizer( sublinear_tf=True, use_idf =True,analyzer='word',ngram_range=(1,2))

    train_sent1=vectorizer1.fit_transform(train_data_mat[:,0])
    train_sent2=vectorizer2.fit_transform(train_data_mat[:,1])
    test_sent1=vectorizer1.transform(test_data_mat[:,0])
    test_sent2=vectorizer2.transform(test_data_mat[:,1])
    test_data=hstack((test_sent1,test_sent2))
    y_gt=test_labels
    y_pred_label = []   
    y_pred_label=model.predict(test_data)
    
    return y_gt,y_pred_label    

    # Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences1, seq_len):
    features = np.zeros((len(sentences1), seq_len),dtype=int)
    for i in range(len(sentences1)):
        try:
            sent_len=len(sentences1[i])
            
            if sent_len!=seq_len:
                if sent_len<seq_len:
                    # print(np.array(sentences[i]+[0 for j in range(seq_len-sent_len)]))
                    features[i]=np.array(sentences1[i]+[vocab_size -1 for j in range(seq_len-sent_len)])
                else:
                    features[i]=np.array(sentences1[i][:seq_len]  )
            else:
                features[i]=np.array(sentences1[i])
        except:
            print(sentences1[i])
            pass
    return features    
def preprocess(d2,pretrained_model):
    global vocab_size
    test_sent1=d2.sentence1
    test_sent2=d2.sentence2
    test_sentences_inp=[i for i in range(test_sent1.shape[0])]
    test_sentences_out=[i for i in range(test_sent2.shape[0])]
    for i, sentence in enumerate(test_sent1):
        sentence=sentence.lower()
        sentence=re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]','',sentence)
        test_sentences_inp[i] = [pretrained_model.wv.vocab.get(word).index if word in pretrained_model.wv.vocab.keys() else vocab_size-1 for word in sentence.split() ]
        # test_sentences_inp[i].append(vocab_size-1)
    for i, sentence in enumerate(test_sent2):
        sentence=sentence.lower()
        sentence=re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]','',sentence)
        # Looking up the mapping dictionary and assigning the index to the respective words
        test_sentences_out[i] = [pretrained_model.wv.vocab.get(word).index if word in pretrained_model.wv.vocab.keys() else vocab_size-1 for word in sentence.split() ]
        # test_sentences_out[i].append(vocab_size-1)
    return test_sentences_inp,test_sentences_out    
    
def main():
    cwd=os.getcwd()
    test_file= cwd+r"/snli_1.0/snli_1.0/snli_1.0_test.csv"
    train_file=cwd+r"/snli_1.0/snli_1.0/snli_1.0_train.csv"
    d1=pd.read_csv(train_file)
    d2=pd.read_csv(test_file)

    glove_file = datapath(cwd+'//vectors_2.txt')
    tmp_file = get_tmpfile(cwd+"/test_word2vec_300_2.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    pretrained_model = KeyedVectors.load_word2vec_format(tmp_file)
    vocab_size=len(pretrained_model.wv.vocab)

    test_sentences_inp,test_sentences_out=preprocess(d2,pretrained_model)
    
    global maxlen
    seq_len = maxlen  # The length that the sentences will be padded/shortened to
    
    
    test_sentences_inp1 = pad_input(test_sentences_inp, seq_len)
    test_sentences_out1 = pad_input(test_sentences_out, seq_len)
    
    mapping={}
    mapping['neutral']=0
    mapping['contradiction']=1
    mapping['entailment']=2
    test_labels_1=[d2.label[i] for i in range(len(d2.label))]
    test_labels=[mapping[d2.label[i]] for i in range(len(d2.label))]
    test_labels=np.array(test_labels,dtype=float)
    
    batch_size = 128
    dataset = TensorDataset(torch.from_numpy(test_sentences_inp1),torch.from_numpy(test_sentences_out1), torch.from_numpy(test_labels))
    testloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Hyperparams
    output_size = 3
    embedding_dim = 300
    vocab_size=len(pretrained_model.wv.vocab)
    hidden_dim = 512
    n_layers = 3

    model_options = {
                  'vocab_size': vocab_size, 
                  'embed_dim': embedding_dim,
                  'out_dim': output_size,
                  'dp_ratio':0.5,
                  'd_hidden':hidden_dim
                }        
    model = LSTM_2(model_options,pretrained_model)
    model.load_state_dict(torch.load("./LSTM_c_best_v.pt" ))

    # from training_conv_net import LeNet
    clf = LogisticRegression(random_state=0,max_iter=10000,multi_class='ovr')
    filename="./Logistic_model.sav"
    clf = pickle.load(open(filename, 'rb'))
    loss, gt, pred = test_deep(model, testloader)
    with open("deepmodel.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    # gen_graph(gt,pred)    
    
    gt, pred = test_logistic(clf, d1,d2,test_labels_1)
    with open("tfidf.txt", 'w') as f:
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")   
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
    # gen_graph(gt,pred)
    
if __name__ == "__main__":
    main()
