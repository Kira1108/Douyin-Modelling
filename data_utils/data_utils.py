import pandas as pd
import numpy as np
from collections import OrderedDict
from collections import Counter
import json
import pymysql
import re
import jieba



def fill_dummy(df):
    '''
        Fill number columns of df with column mean
        Fill object columns of df with string 'Missing'
    '''
    
    num_df = df.select_dtypes(include = ['number']).fillna('mean')
    obj_df = df.select_dtypes('object').fillna('Missing')
    return pd.concat([num_df, obj_df],axis = 1)


def onehot_encoder(arr, prefix = ''):
    '''
        One hot encoder
        ---------------
        Parameters:
            arr: input array like
            predix: encoded class prefix string
            
        Returns:
            one hot encoded dataframe
            
    
    '''
    if (prefix == '') and hasattr(arr,'name'):
        prefix = arr.name

    id2cat = {i:c for i, c in enumerate(set(arr))}
    cat2id = {v:k for k,v in id2cat.items()}
    categorical_values = [cat2id[val] for val in arr]
    onehot_features = np.eye(len(id2cat))[categorical_values]
    return pd.DataFrame(onehot_features, columns = [prefix + '_' + c for c in cat2id.keys()])


def json_to_vec(s):
    '''
    Get vector: 
    '[{"name": "<=17岁", "value": 3.05}, 
      {"name": "18-24岁", "value": 4.46}, 
      {"name": "25-32岁", "value": 2}, 
      {"name": "33-39岁", "value": 7.63}, 
      {"name": "40-46岁", "value": 11.62}, 
      {"name": ">46岁", "value": 15.26}]'
    
    '''
    vector = OrderedDict()
    for dic in json.loads(s):
        vector[dic['name']] = dic['value']
    return np.array(list(vector.values()))


def unique_keys(s):
    '''
    Get key names: 
    '[{"name": "<=17岁", "value": 3.05}, 
      {"name": "18-24岁", "value": 4.46}, 
      {"name": "25-32岁", "value": 2}, 
      {"name": "33-39岁", "value": 7.63}, 
      {"name": "40-46岁", "value": 11.62}, 
      {"name": ">46岁", "value": 15.26}]'
    '''
    return [dic['name'] for dic in json.loads(s)]


def vectorize_json_fixed_length(base_user_fans, col = 'male_age_rate'):
    
    # get feature names
    feature_names = []
    loop_idx = 0
    while len(feature_names) <=0:
        feature_names = unique_keys(base_user_fans[col][loop_idx])
        loop_idx += 1
    
    # vectorization
    vectors = base_user_fans[col].apply(json_to_vec).reset_index(drop = True)
    
    # fill missings
    res = []
    for l in vectors.values:
        if len(l) == 0:
            res.append(np.array([0.] *  len(feature_names)))
        else:
            res.append(l)
    res = np.array(res)
    
    return res, feature_names


def vectorize_json_variable_length(base_user_fans,col = 'comment_cloud', max_keys = None, limit = None):
    
    # find unique key names
    unique_names = []
    for record in base_user_fans[col].values.tolist():
        for dic in json.loads(record):
            unique_names.append(dic['name'])

    key_names = [key for key, count in Counter(unique_names).most_common()]
    
    # filter most common words, make no sense
    if max_keys is not None:
        key_names = key_names[:max_keys]
    
    # mapping tables
    word2index = {word:idx for idx,word in enumerate(key_names)}
    index2word =  {v:k for k,v in word2index.items()}


    result = []
    # for each data point
    for record in base_user_fans[col].values.tolist():

        # there is a vector
        vector = np.zeros(len(word2index))
        r = json.loads(record)

        if (len(r) >0) and (limit is not None):
            r = r[:limit]

        # constructed from dictionary
        for dic in r:
            vector[word2index.get(dic['name'],0)] = dic['value']
        result.append(vector)


    return np.array(result), word2index


def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def replace_puncs(x):
    return re.sub(r'[^\w\s]','',x)

def remove_blanks(x):
    return re.sub(r'\s{2,}','',x)

def cut_text(x):
    return list(jieba.cut(x,cut_all=False))