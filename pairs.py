import pandas as pd
import numpy as np
import os
from pystocktwits.utils import return_json_file
import time
from datetime import datetime,timedelta
from sklearn import preprocessing
from sklearn.linear_model import Perceptron

import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

import joblib
import json
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import sys

tweetpath = './dataset/tweet/preprocessed/'
pricepath = './dataset/price/raw/'

StockList= os.listdir(tweetpath)
StockList.sort()

PriceList= os.listdir(pricepath)
PriceList.sort()

window_days = 1

seed_val = 2021

tech = ['GOOG', 'MSFT', 'FB', 'T', 'CHL', 'ORCL', 'TSM', 'VZ', 'INTC', 'CSCO']
fin = ['BCH', 'BSAC', 'BRK-A', 'JPM', 'WFC', 'BAC', 'V', 'C', 'HSBC', 'MA']
all_stocks = tech + fin

stock1 = sys.argv[1]
stock2 = sys.argv[2]

from pystocktwits import Streamer

twit = Streamer()

# Get User Messages by ID
raw_json = twit.get_user_msgs("170", since=0, max=1, limit=1)

return_json_file(raw_json, 'result.json')

with open('result.json', "r") as data_file:
    data = pd.DataFrame(data_file)
    
def Gendf_stock(day, stockname):
    newstr = []
    
    path = tweetpath + stockname
    files= os.listdir(path)

    for file in files:
        if datetime.strptime(file, "%Y-%m-%d") == day:
            all_data = [json.loads(line) for line in open(path+"/"+file, 'r')]
            for each_dictionary in all_data:
                text = each_dictionary['text']
                newstr += text
                newstr += '\n'
            break    
                
    return [newstr]

def get_df(stockname):
    
    price = pd.read_csv(pricepath + stockname + '.csv')
    
    for i in range(len(price)):
        price['Date'][i] = price['Date'][i].replace('-', '')
        
    price['Strength'] = 0.0

    for i in range(len(price) - window_days):
        price['Strength'][i] = float(price['Close'][i + window_days] - price['Open'][i + window_days]) / (price['Open'][i + window_days])

    price = price.dropna().reset_index(drop=True)
    
    day = datetime(2014, 1, 1)
    end = datetime(2016, 1, 1)

    tw = pd.DataFrame()

    while day < end:
        temptw = pd.DataFrame()
        temptw['content'] = Gendf_stock(day, stockname)
        temptw['Date'] = day.strftime("%Y%m%d")
        tw = pd.concat([tw, temptw]) 
        day = day + timedelta(days=1)

    tw = tw.reset_index(drop = True)
    
    tw['Text'] = 0

    for i in range(len(tw)):
        tw_str = str()
        for char in tw['content'][i]:
            tw_str += char + ' '
        tw['Text'][i] = tw_str
        tw['Text'][i] = tw['Text'][i].replace('$', '')
        tw['Text'][i] = tw['Text'][i].replace('URL', '')
        tw['Text'][i] = tw['Text'][i].replace('rt', '')
        tw['Text'][i] = tw['Text'][i].replace('AT_USER', '')
        tw['Text'][i] = tw['Text'][i].replace('->', '')
        tw['Text'][i] = tw['Text'][i].replace('@', '')

    tw = tw[['Date', 'Text']]

    df = pd.merge(tw, price, on = 'Date', how = 'left')
    df = df[['Date', 'Text', 'Strength']]
    df = df.dropna()
    df = df.reset_index(drop=True)

    for i in range(len(df)):
        df['Text'].iloc[i] = df['Text'][i].split('\n')
        
    df['sentence_embeddings'] = 0
    df['sentence_embedding'] = 0
    df = df.astype('object')

    for i in range(len(df)):
        df['sentence_embeddings'].iloc[i] = model.encode(df['Text'][i])
        df['sentence_embedding'].iloc[i] = df['sentence_embeddings'].iloc[i].mean(axis=0)
    df = df[['Date', 'Strength', 'sentence_embedding']]
    
    return df
 
result = pd.DataFrame()
    
df_stock1 = get_df(stock1)
df_stock2 = get_df(stock2)

df = pd.merge(df_stock1, df_stock2, on = 'Date', how = 'left')
df = df.dropna()

len_df = len(df)

df_train = df[:int(len_df * 0.8)]
df_dev = df[int(len_df * 0.8):int(len_df * 0.9)]
df_test = df[int(len_df * 0.9):]


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
        #print(df[i]['body'])
        sentences.append(df[i]['body'])

    sentence_embeddings = model.encode(sentences)
    sentence_embedding = sentence_embeddings.mean(axis=0)

    return sentence_embedding

print('crawling', stock1, 'tweets ...')
sentence_embedding1 = get_sentence_embedding(df_stock1)

print('crawling', stock2, 'tweets ...')
sentence_embedding2 = get_sentence_embedding(df_stock2)

sentence_embedding = np.append(sentence_embedding1, sentence_embedding2)

def preprocess(df):

    data = pd.DataFrame()
    data = pd.concat([pd.DataFrame(df['sentence_embedding_x'].tolist()), pd.DataFrame(df['sentence_embedding_y'].tolist())], axis = 1)
    data = data.T.reset_index(drop=True)
    data = data.T

    return data, df['Strength_x'] > df['Strength_y']

X_train, y_train = preprocess(df_train)
X_dev, y_dev = preprocess(df_dev)
X_test, y_test = preprocess(df_test)

clf = Perceptron(tol=1e-3, random_state=seed_val)
clf.fit(X_train, y_train)

def report_acc(df_test, clf):

    pred_test = clf.predict(X_test)

    acc = round(accuracy_score(y_test, pred_test) *100, 2)

    print('\nHistorical Testing Accuracy : ', acc , '%')

    return acc

acc = report_acc(df_test, clf)

pred = clf.predict(sentence_embedding.reshape(1, -1))

def compute_profit_risk(df, X_):

    df['y_true'] = df['Strength_x'] > df['Strength_y']
    y_pred = clf.predict(X_)
    df['y_pred'] = y_pred

    df['profit'] = 0.0

    for i in range(len(df)):
        if df['y_pred'].iloc[i] == True:
            df['profit'].iloc[i] =  df['Strength_x'].iloc[i] - df['Strength_y'].iloc[i]
        else:
            df['profit'].iloc[i] = - df['Strength_x'].iloc[i] + df['Strength_y'].iloc[i]

    total_return = final_return = 10000
    for i in range(len(df)):
        final_return = final_return + df['profit'].iloc[i] * final_return

    ret = round((final_return - total_return) / total_return * 100, 2)
    risk = round(df['profit'].std() *100, 2)

    print('Historical return on investment : ', ret, '%')
    print('Historical investment risk : ', risk, '%\n')

    return ret, risk

ret, risk = compute_profit_risk(df_test, X_test)


print('Decision : ')
if pred == True:
    output_result = 'Long ' + stock1 + ' Short ' + stock2
    print(output_result)

elif pred == False:
    output_result = 'Long ' + stock2 + ' Short ' + stock1
    print(output_result)

result['decision'] = [output_result]

result['acc'] = [acc]
result['return'] = [ret]
result['risk'] = [risk]

result.index = ['report']

result.to_json('report/' + stock1 + stock2 + '.json', orient="records")
result.to_json('report/' + stock2 + stock1 + '.json', orient="records")
    