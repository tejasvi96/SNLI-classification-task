from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import os
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from collections import Counter
import nltk
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
import numpy as np
from torch.utils.data import RandomSampler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
root_path=os.getcwd()
cwd=os.getcwd()
 

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
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
#     self.hidden=self.init_hidden()
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
    encoded_premise, _ = self.lstm(premise_proj)
    encoded_hypothesis, _ = self.lstm(hypothesis_proj)
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

  def LSTM_2(options):
    return LSTM_2(options)

def load_data():
    cwd=os.getcwd()
    TRAIN_PATH = cwd+r'/snli_1.0/snli_1.0/snli_1.0_train.txt'
    DEV_PATH = cwd+'/snli_1.0/snli_1.0/snli_1.0_dev.txt'
    TEST_PATH = cwd+'/snli_1.0/snli_1.0/snli_1.0_test.txt'

    train_df = pd.read_csv(TRAIN_PATH, sep='\t', keep_default_na=False)
    dev_df = pd.read_csv(DEV_PATH, sep='\t', keep_default_na=False)
    test_df = pd.read_csv(TEST_PATH, sep='\t', keep_default_na=False)

    print(f'Number of train, dev and test examples:',len(train_df),len(dev_df),len(test_df))

    def df_to_list(df):
        return list(zip(df['sentence1'], df['sentence2'], df['gold_label']))

    train_data = df_to_list(train_df)
    dev_data = df_to_list(dev_df)
    test_data = df_to_list(test_df)

    def filter_no_consensus(data):
        return [(sent1, sent2, label) for (sent1, sent2, label) in data if label != '-']

    print(f'Examples before filtering:',len(train_data), len(dev_data), len(test_data))
    train_data = filter_no_consensus(train_data)
    dev_data = filter_no_consensus(dev_data)
    test_data = filter_no_consensus(test_data)
    print(f'Examples after filtering:',len(train_data), len(dev_data), len(test_data))

    import spacy

    nlp = spacy.load('en_core_web_sm')

    example_sentence = train_data[12345][0]

    print(f'Before tokenization: {example_sentence}')

    tokenized_sentence = [token.text for token in nlp(example_sentence)]

    print(f'Tokenized: {tokenized_sentence}')

    from tqdm import tqdm

    def tokenize(string):
        return ' '.join([token.text for token in nlp.tokenizer(string)])


    def tokenize_data(data):
        return [(tokenize(sent1), tokenize(sent2), label) for (sent1, sent2, label) in tqdm(data)]

    train_data = tokenize_data(train_data)
    dev_data = tokenize_data(dev_data)
    test_data = tokenize_data(test_data)

    train_df = pd.DataFrame.from_records(train_data)
    dev_df = pd.DataFrame.from_records(dev_data)
    test_df = pd.DataFrame.from_records(test_data)

    headers = ['sentence1', 'sentence2', 'label']

    train_df.to_csv(f'{TRAIN_PATH[:-4]}.csv', index=False, header=headers)
    dev_df.to_csv(f'{DEV_PATH[:-4]}.csv', index=False, header=headers)
    test_df.to_csv(f'{TEST_PATH[:-4]}.csv', index=False, header=headers)


def datafunc(d1,d2,pretrained_model):
    train_sent1=d1.sentence1
    train_sent2=d1.sentence2
    test_sent1=d2.sentence1
    test_sent2=d2.sentence2


    train_sentences_inp=[i for i in range(train_sent1.shape[0])]
    train_sentences_out=[i for i in range(train_sent2.shape[0])]


    vocab_size=len(pretrained_model.wv.vocab)

    for i, sentence in enumerate(train_sent1):
        sentence=sentence.lower()
        sentence=re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]','',sentence)
        train_sentences_inp[i] = [pretrained_model.wv.vocab.get(word).index  for word in sentence.split() if word in pretrained_model.wv.vocab.keys() ]
    for i, sentence in enumerate(train_sent2):
        sentence=sentence.lower()
        sentence=re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]','',sentence)
            # Looking up the mapping dictionary and assigning the index to the respective words
        train_sentences_out[i] = [pretrained_model.wv.vocab.get(word).index    for word in sentence.split() if word in pretrained_model.wv.vocab.keys() ]

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
        test_sentences_out[i] = [pretrained_model.wv.vocab.get(word).index if word in pretrained_model.wv.vocab.keys() else vocab_size-1 for word in sentence.split() ]
    return train_sentences_inp,train_sentences_out,test_sentences_inp,test_sentences_out

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
#         print(sentences1[i])
        pass
    return features

  # The length that the sentences will be padded/shortened to



def preprocess(train_sentences_inp,train_sentences_out,test_sentences_inp,test_sentences_out):
    maxlen=-20
    for i in train_sentences_inp:
      if maxlen<len(i):
        maxlen=len(i)  
    for i in train_sentences_out:
      if maxlen<len(i):
        maxlen=len(i)  
    seq_len=maxlen
    train_sentences_inp1 = pad_input(train_sentences_inp, seq_len)
    test_sentences_inp1 = pad_input(test_sentences_inp, seq_len)

    train_sentences_out1 = pad_input(train_sentences_out, seq_len)
    test_sentences_out1 = pad_input(test_sentences_out, seq_len)
    return(train_sentences_inp1,train_sentences_out1,test_sentences_inp1,test_sentences_out1)

# Defining a function that either shortens sentences or pads sentences with unknown token to a fixed length




def datasetfunc(batch_size,train_sentences_inp1,train_sentences_out1,test_sentences_inp1,test_sentences_out1,train_labels,test_labels):
    split_frac = 0.5 
    split_id = int(split_frac * len(test_sentences_inp1))
    print(len(test_sentences_inp1),split_id)
    val_sentences_inp1, test_sentences_inp1 = test_sentences_inp1[:split_id], test_sentences_inp1[split_id:]
    val_sentences_out1, test_sentences_out1 = test_sentences_out1[:split_id], test_sentences_out1[split_id:]

    val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]


    # TensorDataset??
    train_data1 = TensorDataset(torch.from_numpy(train_sentences_inp1),torch.from_numpy(train_sentences_out1), torch.from_numpy(train_labels))
    val_data1 = TensorDataset(torch.from_numpy(val_sentences_inp1),torch.from_numpy(val_sentences_out1), torch.from_numpy(val_labels))
    test_data1 = TensorDataset(torch.from_numpy(test_sentences_inp1),torch.from_numpy(test_sentences_out1), torch.from_numpy(test_labels))


    train_loader = DataLoader(train_data1, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data1, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data1, shuffle=True, batch_size=batch_size)
    return(train_loader,val_loader,test_loader)




# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
def model_init(output_size,embedding_dim,lr,batch_size,n_layers,hidden_dim,vocab_size,pretrained_model):
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset_options = {
                  'batch_size': batch_size, 
                  'device': device
                }

    model_options = {
                  'vocab_size': vocab_size, 
                  'embed_dim': embedding_dim,
                  'out_dim': output_size,
                  'dp_ratio':0.5,
                  'd_hidden':hidden_dim
                }           
    model = LSTM_2(model_options,pretrained_model)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr = lr,weight_decay=1e-4)

    model.to(device)
    opt_file = 'optim_c_best.pth'
    #path = F"/content/drive/My Drive/{opt_file}" 
    #opt.load_state_dict(torch.load(path))
    print(model)
    return model

#training part
def train_model(model,train_loader,val_loader,batch_size,epochs):
    model.train()
    maxval=0
    from torch.utils.data import RandomSampler
    model.train()
    # epochs=100
    for i in range(epochs):
      n_correct, n_total, n_loss = 0, 0, 0
      train_sampler = RandomSampler(train_data1)
      train_loader = torch.utils.data.DataLoader(train_data1, batch_size=batch_size,sampler=train_sampler)
      for batch_idx, (premise,hypothesis,label) in enumerate(train_loader):
        opt.zero_grad()
        premise=premise.to(device)
        hypothesis=hypothesis.to(device)
        label=label.to(device)
        answer = model(premise,hypothesis)
        loss = criterion(answer, label.long())

        n_correct += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
        n_total += batch_size
        n_loss += loss.item()

        loss.backward(); 
        opt.step()
      val_acc=val_loss(model,val_loader)
      if val_acc>maxval:
        maxval=val_acc
        model_file = 'LSTM_c_best_v.pt'
        path = F"/content/drive/My Drive/{model_file}" 
        torch.save(model.state_dict(),path)
        opt_file = 'optim_c_best_v.pth'
        path = F"/content/drive/My Drive/{opt_file}" 
        torch.save(opt.state_dict(), path)

      model_file = 'LSTM_c_best_t.pt'
      path = F"/content/drive/My Drive/{model_file}" 
      torch.save(model.state_dict(),path)
      opt_file = 'optim_c_best_t.pth'
      path = F"/content/drive/My Drive/{opt_file}" 
      torch.save(opt.state_dict(), path)
      train_loss = n_loss/n_total
      train_acc = 100. * n_correct/n_total
      print(i,train_loss, train_acc)
    return model



# import seaborn as sn
# import matplotlib.pyplot as plt
# xticklabels=['neutral','contradiction','entailment']
# fig = sn.heatmap(confusion_matrix, xticklabels=xticklabels,yticklabels=xticklabels,annot=True)
# plt.title("Confusion Matrix for Deep Model")
# plt.show()


def val_loss(model,loader):
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    n_correct, n_total, n_loss = 0, 0, 0
    confusion_matrix = torch.zeros(3,3)
    with torch.no_grad():
      for batch_idx, (premise,hypothesis,target) in enumerate(loader):
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

      test_loss = n_loss/n_total
      test_acc = 100. * n_correct/n_total
    print(test_loss, test_acc, confusion_matrix)
    return(test_acc)
    





def main():
    load_data()
    glove_file = datapath(cwd+'//vectors_2.txt')
    tmp_file = get_tmpfile(cwd+"/test_word2vec_300_2.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    pretrained_model = KeyedVectors.load_word2vec_format(tmp_file)
    train_file = cwd+r"/snli_1.0/snli_1.0/snli_1.0_train.csv"
    test_file= cwd+r"/snli_1.0/snli_1.0/snli_1.0_test.csv"
    d1=pd.read_csv(train_file)
    d2=pd.read_csv(test_file)
    vocab_size=len(pretrained_model.wv.vocab)
    mapping={}
    mapping['neutral']=0
    mapping['contradiction']=1
    mapping['entailment']=2
    
    train_labels=np.array([mapping[d1.label[i]] for i in range(len(d1.label))])
    test_labels=np.array([mapping[d2.label[i]] for i in range(len(d2.label))])
    train_sentences_inp,train_sentences_out,test_sentences_inp,test_sentences_out=datafunc(d1,d2,pretrained_model)
    train_sentences_inp1,train_sentences_out1,test_sentences_inp1,test_sentences_out1=preprocess(train_sentences_inp,train_sentences_out,test_sentences_inp,test_sentences_out)
    batch_size=128
    train_loader,val_loader,test_loader=datasetfunc(batch_size,train_sentences_inp1,train_sentences_out1,test_sentences_inp1,test_sentences_out1,train_labels,test_labels)
    output_size = 3
    embedding_dim = 300
    lr=0.0005

    hidden_dim = 512
    n_layers = 3
    
    model = model_init(output_size,embedding_dim,lr,batch_size,n_layers,hidden_dim,vocab_size,pretrained_model)
    epochs=0
    model=train_model(model,train_loader,val_loader,batch_size,epochs)
    model.load_state_dict(torch.load(r"LSTM_c_best_v.pt" ))
if __name__=='__main__':
    main()  


