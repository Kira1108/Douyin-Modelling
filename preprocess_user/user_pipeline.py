from etl import load_user_info_features
from etl import load_fans_info_features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion



def make_user_fans_pipeline(n_regions = 30):

    '''
        How to use a pipeline,
        ```
            user = load_user_info_features()
            fans = load_fans_info_features()
            df = user.merge(fans, on = 'id', how = 'inner')
            pipe = make_user_fans_pipeline(30)
            transformed_data pipe.fit_transform(df)
        ```
    '''

    # define numeric features
    numeric_cols = ['fans','videos','index_number','likes','day_周一','day_周二', 'day_周三', 'day_周四','day_周五',
     'day_周六','day_周日','week_0','week_1','week_2','week_3','week_4','week_5','week_6','week_7','week_8','week_9',
     'week_10','week_11','week_12','week_13','week_14','week_15','week_16','week_17','week_18','week_19','week_20',
     'week_21','week_22','week_23','male_<=17岁','male_18-24岁','male_25-32岁','male_33-39岁','male_40-46岁','male_>46岁',
     'female_<=17岁','female_18-24岁','female_25-32岁','female_33-39岁','female_40-46岁','female_>46岁',
     'city_level_其他','city_level_超一线与一线','city_level_三线','city_level_二线','city_level_四线','city_level_五线']

    # define categorical features
    cat_cols = ['sex_未知','sex_男','sex_女','tag_职场','tag_明星','tag_政务','tag_情感',
     'tag_其他','tag_搭配','tag_动漫','tag_帅哥','tag_食品','tag_宠物','tag_剧情',
     'tag_教育','tag_科技','tag_蓝V','tag_舞蹈','tag_知识','tag_幽默','tag_游戏',
     'tag_汽车','tag_生活','tag_母婴','tag_奇趣','tag_教学','tag_美女','tag_种草',
     'tag_运动','tag_美妆','tag_家居','tag_音乐','tag_品味','tag_健康','tag_少儿',
     'tag_影视','tag_旅行','city_level_Others','city_level_五线城市','city_level_一线城市',
     'city_level_四线城市','city_level_三线城市','city_level_Missing','city_level_二线城市']

    # numeric pipeline
    num_selector = FunctionTransformer(lambda x:x[numeric_cols])
    log_transformer = FunctionTransformer(lambda x:np.log1p(x))
    scaler = StandardScaler()
    numeric_pipeline = Pipeline(steps = [('num_select', num_selector),
                                         ('log_transform',log_transformer),
                                         ('scaler', scaler)])

    # compress region features
    def region_feature_compressor(df,n = n_regions):
        region_cols = [col for col in df.columns if col.startswith('region')]
        top_regions = df[region_cols].sum().\
            sort_values(ascending = False).iloc[1:n].index.tolist()
        region_df = df[top_regions].copy()
        region_df['region_others'] = 0
        region_df.loc[(region_df.sum(axis = 1) == 0).values,'region_others'] = 1
        return region_df
    region_compressor = FunctionTransformer(region_feature_compressor)

    # categorical features
    categorical_selector = FunctionTransformer(lambda x:x[cat_cols])

    # make a pipeline
    user_fans_pipeline = FeatureUnion([('numeric_pipe', numeric_pipeline),
                                   ('region_features',region_compressor),
                                   ('categorical',categorical_selector)])

    return user_fans_pipeline
