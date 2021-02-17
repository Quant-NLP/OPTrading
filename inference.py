import pandas as pd
import numpy as np
from pystocktwits.utils import return_json_file

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

import joblib
import sys

stock1 = sys.argv[1]
stock2 = sys.argv[2]

from pystocktwits import Streamer

twit = Streamer()

# Get User Messages by ID
raw_json = twit.get_user_msgs("170", since=0, max=1, limit=1)

return_json_file(raw_json, 'result.json')

with open('result.json', "r") as data_file:
    data = pd.DataFrame(data_file)
    
def get_stocktwit(stock):
    raw_json = twit.get_symbol_msgs(symbol_id=stock, since=0, max=0, limit=30, callback=None, filter=None)
    #return_json_file(raw_json, 'result.json')
    df = pd.json_normalize(raw_json)['messages'][0]
    return df

df_stock1 = get_stocktwit(stock1)
df_stock2 = get_stocktwit(stock2)

def get_sentence_embedding(df):
    sentences = []

    for i in range(len(df)):
        sentences.append(df[i]['body'])
        
    sentence_embeddings = model.encode(sentences)
    sentence_embedding = sentence_embeddings.mean(axis=0)
    
    return sentence_embedding

sentence_embedding1 = get_sentence_embedding(df_stock1)
sentence_embedding2 = get_sentence_embedding(df_stock2)

sentence_embedding = np.append(sentence_embedding1, sentence_embedding2)

clf = joblib.load('OPTrading_LR.pkl')

pred = clf.predict(sentence_embedding.reshape(1, -1))

print('Decision : ')
if pred == True:
    print('Long', stock1, 'Short', stock2)
elif pred == False:
    print('Long', stock2, 'Short', stock1)
    