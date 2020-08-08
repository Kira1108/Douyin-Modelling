import pandas as pd
from data_utils import fill_dummy, onehot_encoder
from functools import reduce
import json
import pymysql
import os


def fill_missing_complex(x):
    '''Customized encoder'''

    if (x is None) or (len(x) == 0):
        return 'Missing'
    else:
        return x

def user_info_features(base_user_info,
                       categorical_features = ['sex','region','tag','city_level']):

    '''
        Base user info processing
        Pass a dataframe and categorical feature names
    '''

    # simple cleaning region names
    res = []
    for r in base_user_info.region:
        if (r is not None) and (r.endswith('市')):
            res.append(r[:-1])
        else:
            res.append(r)
    base_user_info['region'] = res

    # fill missing regions
    base_user_info['region'] = base_user_info['region'].apply(fill_missing_complex)

    # simple cleaning city names
    city_dic = json.load(open(os.path.join('resource','city.txt'),'r', encoding = 'GB18030'))
    city_table = []
    for k,v in city_dic.items():
        for c in v:
            city_table.append([k,c])
    city = pd.DataFrame(city_table, columns = ['city_level','city'])
    base_user_info = base_user_info.merge(city, left_on = 'region',right_on = 'city', how = 'left').drop('city', axis = 1)
    base_user_info.loc[(base_user_info.city_level.isnull()) & (base_user_info.region == 'Missing'),'city_level'] = 'Missing'
    base_user_info.loc[(base_user_info.city_level.isnull()) & (base_user_info.region != 'Missing'),'city_level'] = 'Others'

    # fill other null values
    base_user_info = fill_dummy(base_user_info)


    # categorical onehot encoding
    dfs = []
    for f in categorical_features:
        dfs.append(onehot_encoder(base_user_info[f]))

    onehot_features = reduce(lambda x, y: pd.concat([x, y],axis = 1), dfs)
    numerical_features = ['fans', 'videos','index_number','likes']
    feature_df = pd.concat([onehot_features, base_user_info[numerical_features]],axis = 1)
    feature_df['id']= base_user_info['id'].copy()
    return feature_df


def load_user_info_features(filter_cols = False):
    conn = pymysql.connect(host = '127.0.0.1',
                       user = 'root',
                       password = 'root123',
                       db = 'delidou')

    base_user_info = pd.read_sql_query('select * from base_user_info', conn)
    user_feature_df = user_info_features(base_user_info)
    
    if filter_cols:
        cols = [col for col in user_feature_df.columns if not col.startswith('region')]
        user_feature_df = user_feature_df[cols]
    conn.close()
    return user_feature_df
