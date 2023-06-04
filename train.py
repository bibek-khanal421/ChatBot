import torch
from torchmetrics import Accuracy
from torch.nn import Sequential, Linear, ReLU, Softmax, CrossEntropyLoss
from torch.optim import Adam
import json 
import sys 
import getopt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
import os

#initializing the opts and args variables
opts, args = getopt.getopt(sys.argv[1:], "e:l:b:", ["epoch=", "learning_rate=", "batch_size="])

epoch = 500
learning_rate = 0.01
batch_size = 0

for opt, arg in opts:
    if opt in ("-e", "--epoch"):
        try: 
            epoch = int(arg)
        except ValueError:
            raise ValueError("Epoch must be an integer")
        
    elif opt in ("-l", "--learning_rate"):
        try: 
            learning_rate = float(arg)
        except ValueError:
            raise ValueError("Learning rate must be a float")
        
    elif opt in ("-b", "--batch_size"):
        try: 
            batch_size = int(arg)
        except ValueError:
            raise ValueError("Batch size must be an integer")

class Train():

    def save_params(self, name, value):
        path = 'params/{}.pickle'.format(name)
        with open(path, 'wb') as file:
            pickle.dump({name:value}, file)
            file.close()      

    def get_data(self):
        name2id = {}
        id2name = {}
        data = []
        count = 0
        json_data = json.load(open("intent.json", 'rb'))['intent']
        print("\nLoading the data...")
        for lines in tqdm(json_data):
            if lines["label"] not in name2id:
                name2id[lines["label"]] = count
                id2name[count] = lines["label"]
                data.extend([(sent,count) for sent in lines['pattern']])
                count += 1
        self.save_params('name2id', name2id)
        self.save_params('id2name', id2name)
        return np.array(data), name2id
        
    def process_data(self, data):
        vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords.words('english')).fit([str(i[0]) for i in data])
        X = vectorizer.transform([str(i[0]) for i in data])
        self.save_params('vectorizer', vectorizer)
        X = torch.tensor(np.array(X.toarray(), dtype = np.float32))
        y = torch.tensor(np.array([int(i[1]) for i in data], dtype=np.float32))
        return X,y

    
    def get_model(self, input_size, output_size, hidden_size):
        model = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size),
            Softmax(dim=1)
        )
        return model

    def training(self, X, y, input_size, output_size, hidden_size, batch_size):

        loss_fn = CrossEntropyLoss()
        accuracy_fn = Accuracy(task="multiclass", num_classes = output_size)
        model = self.get_model(input_size, output_size, hidden_size)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        running_loss = []
        running_acc = []
        print("\nTraining the model...")
        for x in tqdm(range(epoch)):
            for i in range(len(y)//batch_size):
                inputs, labels = X[i * batch_size : (i+1) * batch_size].reshape(-1,input_size), y[i * batch_size : (i+1) * batch_size].reshape(-1).type(torch.LongTensor)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                accuracy = accuracy_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                running_acc.append(accuracy)
            # if x % 50 == 0: 
            #     print("Epoch {} loss : {:.3f} accuracy : {:.3f}".format(x, np.mean(running_loss), np.mean(running_acc)))
        print("Training loss : {:.3f} accuracy: {:.3f}".format(np.mean(running_loss), np.mean(running_acc)))
        model_name = 'model.pth'
        save_path = 'models/' + model_name
        torch.save(model, save_path)
        print("\n Model saved at models/{}".format(model_name))
                


def main():
    obj = Train()
    data, index = obj.get_data()
    X,y = obj.process_data(data)
    input_size, output_size = X.shape[1], len(index)
    if batch_size == 0:
        obj.training(X, y, input_size, output_size, input_size//2, batch_size = len(y))
    else:
        obj.training(X, y, input_size, output_size, input_size//2, batch_size = batch_size)

if __name__ == '__main__':
    main()       


