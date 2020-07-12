import pymysql
import pandas as pd
import numpy as np


def generate_negative_samples(positive_df, iterations = 10, n_samples = 10000):

    negative_dfs = []
    for it in range(iterations):
        positive_df['rand1'] = np.random.permutation(positive_df.index.tolist())
        positive_df['rand2'] = np.random.permutation(positive_df.index.tolist())
        negative_samples = positive_df[['id','rand1']].merge(positive_df[['link_id','rand2']],
                                 left_on = 'rand1', right_on = 'rand2', how = 'inner')

        negative_samples = negative_samples.drop(['rand1','rand2'],axis = 1)
        negative_dfs.append(negative_samples)

    negatives = pd.concat(negative_dfs)
    rands = np.random.choice(np.arange(len(negatives)),size = n_samples, replace = False)
    return negatives.iloc[rands,:]

def make_dataset():
    conn = pymysql.connect(host = '127.0.0.1',
                           user = 'root',
                           password = 'root123',
                           db = 'delidou')
    positives = pd.read_sql_query('select id,link_id from base_goods', conn)
    negatives = generate_negative_samples(positives,10,len(positives) * 3)
    positives['target'] = 1
    negatives['target'] = 0
    base_df = pd.concat([positives, negatives])
    conn.close()
    return base_df[['id','link_id','target']]


if __name__ == '__main__':
    base_df = make_dataset()
