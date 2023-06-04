import torch 
import json 
import pickle
import numpy as np 

class Chat():
    def __init__(self):

        self.model = torch.load(r'models/model.pth').eval()
        with open(r'params/vectorizer.pickle', "rb" ) as f:
            self.vectorizer = pickle.load(f)['vectorizer']
        with open(r'params/id2name.pickle', "rb" ) as f:
            self.id2name = pickle.load(f)['id2name']
        with open(r'replies.json', "rb") as f:
            self.replies = json.load(f)

    def get_model(self, path = r'models/model.pth'):
        return self.model 
    
    def process_sent(self,sent):
        X = self.vectorizer.transform([sent])
        X = torch.tensor(np.array(X.toarray(), dtype = np.float32))
        return X 
    
    def get_response(self, prediction):
        for rep_l in self.replies['replies']:
            if rep_l['label'] == prediction:
                return np.random.choice(rep_l['reply'])

    

def main():
    print("\nLoading the ChatBot...")
    print("type 'exit' to exit \n")
    chat = Chat()
    model = chat.get_model()
    while True:
        sent = input('>> ')
        if sent == "exit":
            break
        X = chat.process_sent(sent)
        prediction = chat.id2name[np.argmax(model(X).detach().numpy())]
        reply = chat.get_response(prediction)
        print("<{}> {} \n".format(prediction, reply))

if __name__ == '__main__':
    main()

