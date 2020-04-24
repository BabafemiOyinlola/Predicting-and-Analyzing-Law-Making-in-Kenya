import re
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

months = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06', 'july': '07',\
         'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'}

def get_glove(vocab, dim):
    embeddings_matrix = {}
    embedding_file_path = f'./embeddings/glove.6B.{dim}d.txt'
    with open(embedding_file_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            if word in vocab:
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_matrix[word] = coefs
        return embeddings_matrix
    
def extract_embedding_feature(embeddings_matrix, dimension, title):
    bill = (re.sub('[^a-zA-Z ]+', '', title).strip()).split()
    avg = np.mean([embeddings_matrix[w] for w in bill if w in embeddings_matrix], axis=0)
    return avg

def clean_title(bill_title):
    clean_bill_title = re.sub('[^a-zA-Z ]+', '', bill_title).strip()
    return clean_bill_title

def extract_bill_remark(bill_remark):
    remark = ''
    if 'passed' in str(bill_remark):
        remark = 'passed'
    else: 
        remark = 'not passed'
    return remark

def len_text(text):
    text_length = len([len(i) for i in text.split()])
    return text_length

def title_len(title):
    title = re.sub('[^a-zA-Z]+', '', title)
    return len(title)

def convert_worded_dates(cell):
    date = cell
    cell = cell.strip()
    cell = cell.replace(',', '')
    
    if ' ' in cell:
        items = cell.split(' ')
        if any(x in list(months.keys()) for x in items):
            if items[0][0:2].isdigit():
                date = items[0][0:2] + '/' + months[items[1]] + '/' + items[2]
            else:
                date = '0' + items[0][0:1] + '/' + months[items[1]] + '/' + items[2]
    return date

def format_dates(col):
    col = col.strip()
    
    texts = []
    dates = []
    if ' ' in col:
        items = col.split(' ')
        
        for i in items:
            try:
                dates.append(str(pd.to_datetime(i, dayfirst=True).date()))
            except:
                texts.append(i)
                continue
        dates = sorted(dates)
        return (dates, ' '.join(texts))
    else:
        date = col
        try:
            dates.append(str(pd.to_datetime(str(col), dayfirst=True).date()))
        except:
            texts.append(col)
            pass
        dates = sorted(dates)
        return (dates, ' '.join(texts))
    
def clean(text):
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stopwords and len(word) > 2)
    return text

def preprocess_kenyan_bills(filepath):
    df = pd.read_csv(filepath)
    df = df.applymap(lambda s:s.lower() if type(s) == str else s)
    df['1ST READ'] = df['1ST READ'].apply(lambda x: convert_worded_dates(str(x)))
    df['2ND READ'] = df['2ND READ'].apply(lambda x: convert_worded_dates(str(x)))
    df['3RD READ'] = df['3RD READ'].apply(lambda x: convert_worded_dates(str(x)))
    df['DATED'] = df['DATED'].apply(lambda x: convert_worded_dates(str(x)))
    df['MATURITY'] = df['MATURITY'].apply(lambda x: convert_worded_dates(str(x)))
    
    df['1ST READ'] = df['1ST READ'].apply(lambda x: format_dates(str(x)))
    df['2ND READ'] = df['2ND READ'].apply(lambda x: format_dates(str(x)))
    df['3RD READ'] = df['3RD READ'].apply(lambda x: format_dates(str(x)))

    temp = ['']*len(df)
    df['YEAR'] = temp
    df['MONTH'] = [0]*len(df) #month with 0 did not go into the first reading

    for i, row in df.iterrows():
        if row['DATED'] != 'nan':
            date = row['DATED']
            if len(date.split('/')[1]) > 4:
                date = date.split('/')[0] + '/' + date.split('/')[1][0:2] + '/' + date.split('/')[1][2:]
            elif len(date.split('/')[2]) < 3:
                date = date.split('/')[0] + '/' + date.split('/')[1] + '/20' + date.split('/')[2]
            elif len(date.split('/')[2]) == 3:
                date = date.split('/')[0] + '/' + date.split('/')[1] + '/20' + date.split('/')[2][-2:]
    

            df.at[i,'MONTH'] = datetime.strptime(date, '%d/%m/%Y').month
        try:
            df.at[i,'YEAR'] = pd.to_datetime(row['BILL'][-4:]).year
        except:
            if str(row['1ST READ']) != 'NaT':
                df.at[i,'YEAR'] =  (pd.to_datetime(row['1ST READ'][0]).year)[0]
            else:
                df.at[i,'YEAR'] = pd.to_datetime(row['DATED']).year
                
    df['TEXT'] = df['TEXT'].apply(clean)
    return df