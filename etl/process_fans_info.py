import pandas as pd
from data_utils import *
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer


def fans_info_features():
    conn = pymysql.connect(host = '127.0.0.1',
                       user = 'root',
                       password = 'root123',
                       db = 'delidou')

    base_user_fans = pd.read_sql_query('select * from base_user_fans', conn)


    # data extraction from json
    fans_active_week_rate, fans_awr_names = vectorize_json_fixed_length(base_user_fans, 'fans_active_week_rate')
    male_age_rate, male_ar_names = vectorize_json_fixed_length(base_user_fans, 'male_age_rate')
    female_age_rate, female_ar_names = vectorize_json_fixed_length(base_user_fans, 'female_age_rate')
    fans_active_day_rate, fans_day_names = vectorize_json_fixed_length(base_user_fans, 'fans_active_day_rate')
    word_cloud,word2index = vectorize_json_variable_length(base_user_fans, 'comment_cloud')
    city_rate,city_rate_name = vectorize_json_variable_length(base_user_fans, 'city_rate')
    base_user_fans_small = base_user_fans[['id','create_time','male_rate','female_rate']]
    city_rate_level, city_rate_level_name = vectorize_json_variable_length(base_user_fans,'city_level_rate')

    # concat features
    np_features = np.c_[fans_active_week_rate,
      fans_active_day_rate,
      male_age_rate,
      female_age_rate,
      city_rate_level]

    column_names =  ['day_'+ col for col in fans_awr_names] + \
                    ['week_' + col for col in fans_day_names ] + \
                    ['male_' + col for col in male_ar_names] + \
                    ['female_' + col for col in female_ar_names] + \
                    ['city_level_' + col for col in city_rate_level_name]

    fans_features = pd.DataFrame(np_features, columns=column_names)

    # convert word clound features
    count_features = pd.DataFrame(word_cloud, columns = list(word2index.keys()))
    vectorizer = TfidfTransformer(norm = 'l1')
    vectorizer.fit(word_cloud)
    word_cloud_transformed = vectorizer.transform(word_cloud)
    indices = np.argsort(np.array(word_cloud_transformed.sum(axis = 0))[0])[-500:]
    word_colud_features = count_features.iloc[:,indices]

    return pd.concat([fans_features, word_colud_features,base_user_fans[['id']]],axis = 1)
