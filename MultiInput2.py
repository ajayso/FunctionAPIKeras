# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:59:15 2019

@author: Ajay Solanki
"""
from keras.models import Model
from keras import layers
from keras import Input
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import json
import pandas as pd
import os

class MultiInput:
    def load_data(self):
        data_dir= os.getcwd()
        file_name = os.path.join(data_dir,"news.json")
        data = [json.loads(line) for line in open(file_name, 'r')]
        df = pd.DataFrame(data)
        df = pd.DataFrame(data)
        df["category"] = df["category"].astype('category')
        df["category_cat"] = df["category"].cat.codes
        
        # Load data labels are categories and text is headlines
        labels = []
        texts = []
        labels = df["category_cat"]
        texts = df["headline"]
        description = df["short_description"]
        x_train,x_val,y_train,y_val,embedding_matrix = self.Train_Embeddings(texts,labels)
        x_train2,x_val2,y_train2,y_val2,embedding_matrix2 = self.Train_Embeddings(description,labels)
        print(len(x_train))
        print(len(x_train2))
        self.labels = np.asarray(labels)
        self.texts = texts
        self.text_vocabulary_size = 10000
        self.question_vocabulary_size = 10000
        self.answer_vocabulary_size = 500
        # Data
        self.train_data_text=x_train
        self.train_data_descriptions = x_train2
        self.train_data_labels = y_train
        
        self.val_data_text=x_val
        self.val_data_descriptions = x_val2
        self.val_data_labels = y_val
    
    def Train_Embeddings(self, texts,labels):
        
        data,word_indexer = self.Tokenize(texts)
        maxlen = 1000
        max_words = 20000
        training_samples = 180000
        validation_samples = 20000
        embedding_dim = 50
        indices = np.arange(data.shape[0])
       
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        
        labels = to_categorical(labels)
        
        
        x_train = data[:training_samples]
        y_train = labels[:training_samples]
        
        x_val = data[training_samples: training_samples + validation_samples]
        y_val = labels[training_samples: training_samples + validation_samples]
        
        current_path = os.getcwd()
        BASE_DIR = os.path.dirname( current_path )
        glove_dir = "E:\workdirectory\Code Name Val Halen\DS Sup\DL\Chapter 14\glove.6B"#os.path.join(BASE_DIR, "glove.6B")
        
        embedding_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'),encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coeff = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coeff
        f.close()
        embedding_index = embedding_index
        # Build an embedding matrix that can load into an Embedding Layer
        # Matrix of shape (max_words, embedding_dim)
        
        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in word_indexer.items():
            if i < max_words:
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return x_train,x_val,y_train,y_val,embedding_matrix
    
    
    def Tokenize(self,texts):
        maxlen = 1000
        max_words = 10000
        
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        #print(sequences)
        
        word_indexer = tokenizer.word_index
        
        
        #print("No of words found " % len(word_indexer))
        #print('Found %s unique tokens.' % len(word_indexer))
        
        data = pad_sequences(sequences, maxlen = maxlen)
        
        print("Shape of data tensor", data.shape)
        #print("Shape of label tensor", len(labels))
        
        return data, word_indexer

    def CreateModel2(self):
        embedding_dim=500
        maxlen = 1000
        maxlen = 1000
        max_words = 10000
        
        text_input = Input(shape=(None,), dtype='int32', name='text')
        embedded_text = layers.Embedding(max_words,embedding_dim)(text_input)
        #layers.Embedding(64, self.text_vocabulary_size)(text_input)
        encoded_text = layers.LSTM(32)(embedded_text)
        
        question_input = Input(shape=(None,), dtype='int32', name='description')
        embedded_description = layers.Embedding(max_words,embedding_dim)(question_input)
        encoded_question = layers.LSTM(16)(embedded_description)
        
        
        concatenated = layers.concatenate([encoded_text,encoded_question],axis=1)
        
        category_labels = layers.Dense(41,activation='softmax')(concatenated)
        
        model = Model([text_input, question_input],category_labels)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()
        self.model = model
        
    def CreateModel(self):
        text_input = Input(shape=(None,), dtype='int32', name='text')
        embedded_text = layers.Embedding(64, self.text_vocabulary_size)(text_input)
        encoded_text = layers.LSTM(32)(embedded_text)
        
        question_input = Input(shape=(None,), dtype='int32', name='question')
        embedded_question = layers.Embedding(64, self.question_vocabulary_size)(question_input)
        encoded_question = layers.LSTM(16)(embedded_question)
        
        
        concatenated = layers.concatenate([encoded_text,encoded_question],axis=1)
        
        answers = layers.Dense(self.answer_vocabulary_size,activation='softmax')(concatenated)
        
        model = Model([text_input, question_input],answers)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()
        self.model = model
    
    def ExecuteModel(self):
        maxlen = 1000
        max_words = 10000
        
        text = np.random.randint(1, self.text_vocabulary_size, size= (num_samples, max_length))
        question = np.random.randint(1, self.question_vocabulary_size, size = (num_samples, max_length))
        answers = np.random.randint(0,1, size = (num_samples, self.answer_vocabulary_size))
        self.model.fit([text,question], answers, epochs =100, batch_size =128)
    
    def ExecuteModel2(self):
        num_samples =1000
        max_length = 100
        
        self.model.fit([self.train_data_text,self.train_data_descriptions], self.train_data_labels, epochs =100, batch_size =128)
        

m = MultiInput() 
m.load_data()
m.CreateModel2()
m.ExecuteModel2()