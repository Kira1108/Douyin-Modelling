from data_utils import cut_text,find_chinese,replace_puncs,remove_blanks
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.layers import Input,Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding,concatenate,Flatten
from keras.models import Model
from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import pymysql


def load_tag_prediction_model(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH,num_words,K):

    embedding_matrix = np.load('item_embedding/resources/embedding_matrix.npy')
    embedding_layer = Embedding(num_words,
                        EMBEDDING_DIM,
                        weights = [embedding_matrix],
                        input_length = MAX_SEQUENCE_LENGTH,
                        trainable = False)

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded_sequences = Embedding(input_dim=num_words, output_dim=200, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix],trainable=False)(inputs)
    l_conv01 = Conv1D(128, 3,activation='relu',padding='same')(embedded_sequences)
    l_pool01 = MaxPooling1D(3)(l_conv01)
    l_conv02 = Conv1D(128, 4,activation='relu',padding='same')(embedded_sequences)
    l_pool02 = MaxPooling1D(5)(l_conv02)
    l_conv03 = Conv1D(128, 5,activation='relu',padding='same')(embedded_sequences)
    l_pool03 = MaxPooling1D(5)(l_conv03)
    l_merge = concatenate([l_pool01,l_pool02,l_pool03],axis=1)

    l_cov2 = Conv1D(128, 3, activation='relu',padding='same')(l_merge)
    l_pool2 = MaxPooling1D(10)(l_cov2)
    l_pool2 = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_pool2)
    encoder = Model(inputs, l_dense)

    encoded_vector = encoder(inputs)

    preds = Dense(K, activation='softmax')(encoded_vector)

    model=Model(inputs,preds)
    model.load_weights('item_embedding/resources/item_embedding_model.h5')
    return model

def preprocess_texts(df,**kwargs):
    df['clean_name'] = df.name.apply(remove_blanks).apply(replace_puncs).apply(find_chinese).values
    df['cutted_text'] = df.clean_name.apply(cut_text).values
    df['texts'] = df.cutted_text.apply(lambda x:' '.join(x)).values
    df.loc[:,'sequence_length'] = df.cutted_text.apply(lambda x:len(x)).values
    df = df[(df.sequence_length >0) & (df.sequence_length <= kwargs['MAX_SEQUENCE_LENGTH'])]
    return df


def predict_tag(df):
    model_config = pickle.load(open('item_embedding/resources/model_config.pkl','rb'))
    model = load_tag_prediction_model(**model_config)
    tokenizer = pickle.load(open('item_embedding/resources/tokenizer.pkl','rb'))
    idx_to_cat = pickle.load(open('item_embedding/resources/idx_to_cat.pkl','rb'))

    df = preprocess_texts(df,**model_config)
    sequences = tokenizer.texts_to_sequences(df.texts)
    padded_sequences = pad_sequences(sequences,maxlen = model_config['MAX_SEQUENCE_LENGTH'])
    preds = model.predict(padded_sequences).argmax(axis = 1)
    preds = [idx_to_cat[p] for p in preds]
    df['prediction'] = preds
    return df

def load_encoder_model(EMBEDDING_DIM, MAX_SEQUENCE_LENGTH,num_words,K):

    embedding_matrix = np.load('item_embedding/resources/embedding_matrix.npy')
    embedding_layer = Embedding(num_words,
                        EMBEDDING_DIM,
                        weights = [embedding_matrix],
                        input_length = MAX_SEQUENCE_LENGTH,
                        trainable = False)

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded_sequences = Embedding(input_dim=num_words, output_dim=200, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix],trainable=False)(inputs)
    l_conv01 = Conv1D(128, 3,activation='relu',padding='same')(embedded_sequences)
    l_pool01 = MaxPooling1D(3)(l_conv01)
    l_conv02 = Conv1D(128, 4,activation='relu',padding='same')(embedded_sequences)
    l_pool02 = MaxPooling1D(5)(l_conv02)
    l_conv03 = Conv1D(128, 5,activation='relu',padding='same')(embedded_sequences)
    l_pool03 = MaxPooling1D(5)(l_conv03)
    l_merge = concatenate([l_pool01,l_pool02,l_pool03],axis=1)

    l_cov2 = Conv1D(128, 3, activation='relu',padding='same')(l_merge)
    l_pool2 = MaxPooling1D(10)(l_cov2)
    l_pool2 = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_pool2)
    encoder = Model(inputs, l_dense)
    encoder.load_weights('item_embedding/resources/item_encoding.h5')
    return encoder

def item_encode(df):
    model_config = pickle.load(open('item_embedding/resources/model_config.pkl','rb'))
    encoder = load_encoder_model(**model_config)
    tokenizer = pickle.load(open('item_embedding/resources/tokenizer.pkl','rb'))

    df = preprocess_texts(df,**model_config)
    sequences = tokenizer.texts_to_sequences(df.texts)
    padded_sequences = pad_sequences(sequences,maxlen = model_config['MAX_SEQUENCE_LENGTH'])

    return encoder.predict(padded_sequences)
