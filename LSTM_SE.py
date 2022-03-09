#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:23:41 2020

@author: amouath
"""

import tensorflow as tf
#import tensorflow_addons as tfa
#import keras as keras
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Dense ,BatchNormalization, LSTM, Dropout,RepeatVector,TimeDistributed,Flatten,concatenate,MaxPooling2D,LeakyReLU,Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split as split
#from sklearn.model_selection import LeaveOneOut 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
#from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import csv
import xlrd
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#from sklearn.metrics import roc_curve, auc,average_precision_score
from mlxtend.evaluate import scoring
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
#K.tensorflow_backend._get_available_gpus()
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tensorflow.keras.initializers import RandomNormal,Zeros,Constant

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
#import keras_metrics
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.constraints import max_norm, non_neg,unit_norm

from sklearn.metrics import confusion_matrix
#from tensorflow.keras.utils.vis_utils import plot_model


def open_excel(to_read):

    """
    Open an Excel workbook and read rows from first sheet into sublists
    """
    def read_lines(workbook):
        """ decode strings from each row into unicode lists """
        sheet = workbook.sheet_by_index(0)
        for row in range(sheet.nrows):
            yield [sheet.cell(row, col).value for col in range(sheet.ncols)]
    try:
        workbook = xlrd.open_workbook(to_read)
        return [line for line in read_lines(workbook)]
    except (IOError, ValueError):
        print ("Couldn't read from file %s. Exiting") % (to_read)
        raise




def GET_Label(File_name):
    b=[]
    a=[]
    c=[]
    label=[]
    if (File_name.endswith('SAMM_Micro_FACS_Codes_v2.xlsx')):
        ss = 14
        s = 10
    else:
        ss = 1
        s = 9
    
    L= open_excel(File_name)
    for i in range(ss,len(L)):
        b.append(int(L[i][s])-1)
        a.append(int(L[i][5]))
        c.append(int(L[i][3]))
    
    for j in range(len(a)):
        label.append(b[j])

    return label,b,a,c

def Update_labels(labels):
    LE = LabelEncoder()
    LE.fit(['Anger', 'Disgust', 'Fear', 'Happiness', 'Other','Sadness', 'Surprise'])
    for i in range(len(labels)):
        if labels[i].endswith('Contempt'):
            labels[i] = 'Disgust'
    labels = flatten_vec(labels)
    result = LE.transform(labels)
    print(LE.classes_)
    return result

def arrange_label(lb,File_name): 

    #number of sequence in each subject

    #make the labels as the same form as the data
    l0=[]
    l1=[]
    l2=[]
    l3=[]
    for i in range(6):
        for j in range(len(lb)):
            if i in [0,1]:
                if lb[j] in [1,2,3,4,5,6]:
                    l0.append(lb[j])
                else:
                    l0.append(7)
            elif i in [3,4]:
                if lb[j] in [0,5,6]:
                    l1.append(lb[j])
                else:
                    l1.append(7)
            elif i == 2:
                if lb[j] in [3,5,6] :
                    l2.append(lb[j])
                else:
                    l2.append(7)
            else:
                if lb[j] in [0,1,4,5,6]:
                    l3.append(lb[j])
                else:
                    l3.append(7)
    
    if (File_name.endswith('SAMM_Micro_FACS_Codes_v2.xlsx')):
        h=159
    else:
        h = 255       
    lab = [l0[:h],l0[:h],l2,l1[:h],l1[:h],l3]   
       
    return lab


def flatten_im(s):
    d=[]
    for h in range(len(s)):
        for i in range(len(s[h])):
            if str(type(s[h][i]))[8].endswith('l') :
                for j in range(len(s[h][i])):    
                    d.append(s[h][i][j])
            else:
                d.append(s[h][i])
    print(str(type(s[h][0]))[8])
    return d 

def flatten_vec(v):
    d = np.array([])
    for i in range(len(v)):
        d = np.append(d,v[i])
    return list(d)



def create_RNN_model(R,alpha):
    model = Sequential()
    del model
    
    model = Sequential()
    model.add(LSTM(50,activation='relu',input_shape=(145,alpha[R]*8)))
#    model.add(LSTM(30,activation='relu'))
#    model.add(LSTM(20,activation='relu'))
#    model.add(RepeatVector(1024))
#    model.add(LSTM(20, activation='relu'))
#    model.add(TimeDistributed(Dense(alpha[R]*8)))
    model.add(Dense(20,activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(8,activation='softmax'))
    model.summary()
    return model


def Train_RNN_model(model,xx,yy,R,alpha):
    
    
# it lasts the LOSO Cross validation technic if it takes timpe to split data it is better to do it once for CNN and RNN models    
#    for i,j in enumerate(xx):
#        for k in range(round(145/len(j))):
#            xx[i]+=j

    x1 = pad_sequences(xx,maxlen=145,truncating='post')

    y1=np.array(yy)
    encoder = LabelEncoder()
    encoder.fit(y1)
    encoded_Y = encoder.transform(y1)

    x_train,x_test,y_train,y_test = split(x1,encoded_Y ,test_size=0.2, random_state=42)
    
    class_weights = compute_class_weight('balanced',np.unique(y_train),y_train)
    y_test= to_categorical(y_test,8)
    y_train = to_categorical(y_train,8)
#    x_train = np.array(x_train)
#    x_test = np.array(x_test)
    x_train.reshape(len(x_train),len(x_train[0]),len(x_train[0][0]))
    x_test.reshape(len(x_test),len(x_test[0]),len(x_test[0][0]))
#    l=0
#    x_train_arr =np.array([])
#    x_test_arr= np.array([])
#
#    for i ,j in enumerate(x_train):
#        T = j[l:len(j)][:]
#        x_train[i] = j[l:len(j)][:]
#        l = len(j)
##    for x_t in x_train :    
##        x_train_arr = np.append(x_train_arr,x_t ,axis=0 )    
#    x_train_arr = np.asarray(x_train)
#    l=0
#    
#    for i ,j in enumerate(x_test):
##        T = j[l:len(j)][:]
#        x_test[i] = j[l:len(j)][:]
#        l = len(j)
##        x_test[i] = T
##    for x_tt in x_test :
##        x_test_arr = np.append(x_test_arr,x_tt ,axis=0 ) 
#    x_test_arr = np.asarray(x_test)
#        
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
    history = model.fit(x_train,y_train,validation_data=(x_test, y_test),batch_size=1,epochs =150,class_weight=class_weights)
    model.save('LSTM_R_'+str(R)+'.h5')
    return history

def sparse_loss(y_true, y_pred):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_true)

def adjust_sizes(Data_url):
    # Determine the number of subject and the number of frames
    
    sizes=[] 
    Data=[]
    for R in range(6):
    
        url_0 = Data_url[0][R]
#        url_00 =url_0.split("/")
        f_0 = Image.open(url_0)
        f_0_w,f_0_h=f_0.size
        sizes.append([f_0_h,f_0_w])
        Datas =[]
        for j,x in enumerate(Data_url):
            image = np.array(Image.open(x[R]))
            Datas += [image]
        print(len(Datas))
        Data+=[Datas]
            
    return Data, sizes

_EPSILON = K.epsilon()
def _loss_tensor(y_true, y_pred):
    batch_size =1
#    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    loss =  tf.convert_to_tensor(0,dtype=tf.float32) # initialise the loss variable to zero
    g = tf.constant(1.0, shape=[1], dtype=tf.float32) # set the value for constant 'g'
    for i in range(0,batch_size,1):
        try:
            D = K.sqrt(K.sum((y_pred[i] - y_true[i])**2)) # calculate the euclidean distance between query image and negative image
            loss = (loss + g + D ) # accumulate the loss for each triplet           
        except:
            continue
    loss = loss/batch_size # Average out the loss 
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    return tf.maximum(loss/8000,zero)

def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.
  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.
  Returns:
      A tensor.
  """
  return ops.convert_to_tensor(x, dtype=dtype)

def categorical_crossentropy_modified(target, output, from_logits=False, axis=-1):
  rank = len(output.shape)
  axis = axis % rank
  # Note: nn.softmax_cross_entropy_with_logits_v2
  # expects logits, Keras expects probabilities.
  if not from_logits:
    # scale preds so that the class probas of each sample sum to 1
    #target = target / math_ops.reduce_sum(target, axis, True)
    output = tf.math.multiply(output,target) + tf.math.multiply(1-output,1-target)
    # manual computation of crossentropy
    epsilon_ = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
    return -math_ops.reduce_sum(math_ops.squared_difference(1.0,output)* math_ops.log(output), axis)
  else:
    return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

#class Metrics(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self._data = []
#
#    def on_epoch_end(self, batch, logs={}):
#        X_val, y_val = self.validation_data[0], self.validation_data[1]
#        y_predict = model_c.predict([x_test0,x_test1,x_test2,x_test3,x_test4,x_test5])
#
#        y_val = np.argmax(y_val, axis=1)
#        y_predict = np.argmax(y_predict, axis=1)
#        acc =scoring(y_val,y_predict,metric='per-class accuracy')
#        print('acc = ',acc)
#        self._data.append({
#            'accuracy per class' : acc
##            'precision' : scoring(y_val,y_predict,metric='precision'),
##            'recall' : scoring(y_val,y_predict,metric='recall'),
##            'f1-score' : scoring(y_val,y_predict,metric='f1')
##            'val_recall': recall_score(y_val, y_predict),
##            'val_precision': precision_score(y_val, y_predict),
#        })
#        return
#
#    def get_data(self):
#        return self._data
def flatten(d):
    for i in range(len(d)):
        if i ==0:
            xd =d[i]
        else:
            xd = np.append(xd,d[i],axis=0)
    print('flattened ...')
    return xd
#def flatten(d):
#    s=0
#    for j in range(len(d)):
#        s += len(d[j])
#    xd = np.empty((s,145,d[0].shape[2]))
#    k=0
#    for i in range(len(d)):
#        for j in range(len(d[i])):
#            xd[j+k] =d[i][j]
#        k+=len(d[i])
#    return xd
def sequenced (dataz , yy , dd , R):
    o=[]
    for i in range(len(yy)):
        if yy[i] not in [5,6]:
            o+=[i]
    eee = dd[o[0]][R].split('/')[8][9:]
    sss = dd[o[0]][R].split('/')[7]
    datazz=[]
    for j in range(len(o)):
        ddd = dd[o[j]][R].split('/')[8][9:]
        rrr = dd[o[j]][R].split('/')[7]
        if rrr.endswith(sss):
            if int(ddd) == int(eee):
                if j ==0:
                    dataz_ = [dataz[j]]
                else:
                    dataz_ = np.append(dataz_,[dataz[j]],axis=0)
            else:
                datazz += [dataz_]
                dataz_ = [dataz[j]]
                eee = ddd
        else:
            datazz += [dataz_]
            dataz_ = [dataz[j]]
            eee = ddd
            sss = rrr
            
    datazz += [dataz_]  
    
    return datazz

def fix_label (k, labeln_):
    labelnn = []
    for R in range(6):
        labelnn_ = []
        for j , l  in enumerate(labeln_[R]):
            for i in range(k[j]):
                labelnn_ += [l]
        labelnn += [labelnn_]
    leel = labelnn.copy()
    ooo = []
    for R in range(6):
        oo = []
        for i in range(len(labelnn[R])):
            if labelnn[R][i] in [5,6]:
                oo += [i - len(oo)]
        ooo += [oo]
    
    label_lii =[]
    for R in range(len(ooo)):
        label_li = labelnn[R].copy()
        for j in range(len(ooo[R])):
            label_li.pop(ooo[R][j])
        label_lii += [label_li]
    labelnn = label_lii.copy()
            
    return labelnn,  ooo, leel
def Normalized(data):
    m = data.shape[0]
    sigma =(1/m)*np.sum(data*data,axis=0,keepdims=True)
#    for i in range(sigma.shape[0]):
#        for j in range(sigma.shape[1]):
#            for k in range(sigma.shape[2]):
#                if sigma[i][j][k] == 0:
#                    sigma[i][j][k] = 1
    sigma = sigma + 1
    mu = (1/m)*np.sum(data,axis=0,keepdims=True)
    return mu , sigma
##########################################################################################################
##########################################################################################################
    




'''                         Preparing feature sequence to be the adequate input for the LSTM
'''
                              # ! Begin

                              
    
print("get Image url DATA ...")  
Data_urlx_samm = pd.read_csv('/mnt/DONNEES/Bureau/analyse-ME-DL/Data_url_SAMM.csv',header=None)
Data_url_samm = Data_urlx_samm.values.tolist()   
for i in range(len(Data_url_samm)):
    for j in range(6):
        Data_url_samm[i][j] = Data_url_samm[i][j][:6]+'amouath/Bureau'+Data_url_samm[i][j][14:] 
        
        
Data_urlx_casmeii_ = pd.read_csv('/mnt/DONNEES/Bureau/analyse-ME-DL/Data_url_CASMEII.csv',header=None)   
Data_url_casmeii_ = Data_urlx_casmeii_.values.tolist()   
for i in range(len(Data_url_casmeii_)):
    for j in range(6):
        Data_url_casmeii_[i][j] = Data_url_casmeii_[i][j][:6]+'amouath/Bureau'+Data_url_casmeii_[i][j][14:] 
Data_url_casmeii = []
for i in range(len(Data_url_casmeii_)):
    Data_url_casmeii0 = []
    for j in range(6):
        dc = Data_url_casmeii_[i][j].split('/')
        if (int(dc[7][8:]) != 8 or int(dc[8][9:])!=1) and (int(dc[7][8:]) != 23 or int(dc[8][9:])!=2):
            Data_url_casmeii0.append(Data_url_casmeii_[i][j])
    if len(Data_url_casmeii0) != 0:
        Data_url_casmeii.append(Data_url_casmeii0)
        
        
print("get Label's SAMM (AU) from source ...")   
File_name_s = "/mnt/DONNEES/Bureau/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx"
label_s,b_s,a_s,c_s = GET_Label(File_name_s)
labeln_s = arrange_label(label_s,File_name_s)

print("get Label's CASMEII (AU) from source ...")   
File_name_c = '/mnt/DONNEES/download/CASME2-coding-20190701.xlsx'
label_c,b_c,a_c,c_c = GET_Label(File_name_c)
labeln_c = arrange_label(label_c,File_name_c)


c_s=0
k_s=[]
b_s=[]
t_url_s = Data_url_samm[0][0].split('/')
t_s = t_url_s[8]
r_s = t_url_s[7]
for d  in Data_url_samm:
    dd_s = d[0].split('/')
    s_s = dd_s[7]
    if s_s.endswith(r_s):
        if not dd_s[8].endswith(t_s) :
            k_s += [c_s]
            t_s = dd_s[8]
            c_s = 0
    else:
        k_s += [c_s]
        t_s = dd_s[8]
        r_s = s_s
        c_s = 0
    c_s += 1

k_s += [c_s]

c_c=0
k_c=[]
b_c=[]
t_url_c = Data_url_casmeii[0][0].split('/')
t_c = t_url_c[8]
r_c = t_url_c[7]
for d  in Data_url_casmeii:
    dd_c = d[0].split('/')
    s_c = dd_c[7]
    if s_c.endswith(r_c):
        if not dd_c[8].endswith(t_c) :
            k_c += [c_c]
            t_c = dd_c[8]
            c_c = 0
    else:
        k_c += [c_c]
        t_c = dd_c[8]
        r_c = s_c
        c_c = 0
    c_c += 1

k_c += [c_c]



labelnn_s,  ooo_s, leel_s = fix_label(k_s, labeln_s)
labelnn_c,  ooo_c, leel_c = fix_label(k_c, labeln_c)

for i in reversed(range(len(label_c))):
    if label_c[i] in [5,6]:
        label_c.pop(i)
for i in reversed(range(len(label_s))):
    if label_s[i] in [5,6]:
        label_s.pop(i)
       
################################################## 
##################################################
##################################################
m = 9 # the number of patches; should be in [4,9,16]
fold_sf = 'SF_'+m+'Patches'
Data_url = Data_url_casmeii + Data_url_samm 
leel = []
for i in range(6):
    leel += [leel_c[i] + leel_s[i]]
#leel = [leel_c[0] + leel_s[0]] + [leel_c[1] + leel_c[1]] + [leel_s[2] + leel_c[2]] + [leel_s[3] + leel_c[3]] + [leel_s[4] + leel_c[4]] + [leel_s[5] + leel_c[5]]
print('GET Spatial Features ...')
Data_x0_s = pd.read_csv(fold_sf+'_5_clases_test_'+m+'P_region_0.csv',header=None)
Data_0_s = np.array(Data_x0_s.values.tolist())
Data_S0 = sequenced(Data_0_s,leel[0],Data_url,0)

Data_x1_s = pd.read_csv(fold_sf+'_5_clases_test_'+m+'P_region_1.csv',header=None)
Data_1_s = np.array(Data_x1_s.values.tolist())
Data_S1 = sequenced(Data_1_s,leel[1],Data_url,1)

Data_x2_s = pd.read_csv(fold_sf+'_5_clases_test_'+m+'P_region_2.csv',header=None)
Data_2_s = np.array(Data_x2_s.values.tolist())
Data_S2 = sequenced(Data_2_s,leel[2],Data_url,2)

Data_x3_s = pd.read_csv(fold_sf+'_5_clases_test_'+m+'P_region_3.csv',header=None)
Data_3_s = np.array(Data_x3_s.values.tolist())
Data_S3 = sequenced(Data_3_s,leel[3], Data_url,3)

Data_x4_s = pd.read_csv(fold_sf+'_5_clases_test_'+m+'P_region_4.csv',header=None)
Data_4_s = np.array(Data_x4_s.values.tolist())
Data_S4 = sequenced(Data_4_s,leel[4],Data_url,4)

Data_x5_s = pd.read_csv(fold_sf+'_5_clases_test_'+m+'P_region_5.csv',header=None)
Data_5_s = np.array(Data_x5_s.values.tolist())
Data_S5 = sequenced(Data_5_s,leel[5],Data_url,5)

label_ = label_c +label_s

uuu = Data_url[0][0].split('/')[7][8:]
eee = -1
cal =0
cal_ =[]
for i in range(len(leel[0])):
    if leel[0][i] not in [5,6]:
        ttt = Data_url[i][0].split('/')[7][8:]
        ddd = Data_url[i][0].split('/')[8][9:]
        if int(ttt) == int(uuu):
            if  int(ddd) != int(eee):
                cal+=1
                eee = ddd
        else:
            cal_ += [cal]
            cal = 1
            eee = ddd
            uuu = ttt
cal_+=[cal]


ti = 0     
test0 =[]  
test1 =[]  
test2 =[]  
test3 =[]  
test4 =[]  
test5 =[]  
y_test = []
cm = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
cm_=[]
accuracy=0
accc = []
seq_len = 250
fn = 256
for test_index in range(len(cal_)):
    print('build model ...')
    print("***********************************")
    input0 = Input(shape=(seq_len,fn))
    #input00 = BatchNormalization()(input0)
    lstm0 = LSTM(64,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',return_sequences=False)(input0)  
    lstm0 = LeakyReLU(alpha=0.01)(lstm0)
    lstm0 = Dropout(0.2)(lstm0)
#    lstm0 = LeakyReLU(alpha=0.01)(lstm0)  
    
    input1 = Input(shape=(seq_len,fn))
    #input11 = BatchNormalization()(input1)
    lstm1 = LSTM(64,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',return_sequences=False)(input1)  
    lstm1 = LeakyReLU(alpha=0.01)(lstm1)
    lstm1 = Dropout(0.2)(lstm1)
#    lstm1 = LeakyReLU(alpha=0.01)(lstm1)  
    
    input2 = Input(shape=(seq_len,fn))
    #input22 = BatchNormalization()(input2)
    lstm2 = LSTM(64,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',return_sequences=False)(input2)  
    lstm2 = LeakyReLU(alpha=0.01)(lstm2)
    lstm2 = Dropout(0.2)(lstm2)
#    lstm2 = LeakyReLU(alpha=0.01)(lstm2)  
    
    input3 = Input(shape=(seq_len,fn))
    #input33 = BatchNormalization()(input3)
    lstm3 = LSTM(64,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',return_sequences=False)(input3)  
    lstm3 = LeakyReLU(alpha=0.01)(lstm3)
    lstm3 = Dropout(0.2)(lstm3)
#    lstm3 = LeakyReLU(alpha=0.01)(lstm3)  
    
    input4 = Input(shape=(seq_len,fn))
    #input44 = BatchNormalization()(input4)
    lstm4 = LSTM(64,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',return_sequences=False)(input4)  
    lstm4 = LeakyReLU(alpha=0.01)(lstm4)
    lstm4 = Dropout(0.2)(lstm4)
#    lstm4 = LeakyReLU(alpha=0.01)(lstm4)  
    
    input5 = Input(shape=(seq_len,fn))
    #input55 = BatchNormalization()(input5)
    lstm5 = LSTM(64,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',return_sequences=False)(input5)  
    lstm5 = LeakyReLU(alpha=0.01)(lstm5)
    lstm5 = Dropout(0.2)(lstm5)
#    lstm5 = LeakyReLU(alpha=0.01)(lstm5)  
    
    conc=concatenate([lstm0,lstm1,lstm2,lstm3,lstm4,lstm5]) 
    
# =============================================================================
    feat = int(conc.shape[1])
    l = Dense(int(feat/4),kernel_initializer=RandomNormal(stddev=0.01))(conc)
    l = LeakyReLU(alpha=0.01)(l)
#     l = Dropout(0.5)(l)
    l = Dense(feat,activation='sigmoid')(l)
    l = tf.multiply(conc,l)
# =============================================================================
#    
#    l1 = Dense(1024,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',
#              kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),bias_regularizer=regularizers.l2(1e-5),
#              activity_regularizer=regularizers.l2(1e-6))(l)
#    l1 = LeakyReLU(alpha=0.01)(l1)
#    l1 = Dropout(0.5)(l1)

    l1 = Dense(256,kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros',
               kernel_regularizer=regularizers.l1(1e-5))(l)
    l1 = LeakyReLU(alpha=0.01)(l1)
    l1 = Dropout(0.5)(l1)
#    pred1 = Dense(5, activation='tanh')(l1) 
    
#    l2 = Dense(1024,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',
#              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),bias_regularizer=regularizers.l2(1e-5),
#              activity_regularizer=regularizers.l2(1e-6))(l)
#    l2 = LeakyReLU(alpha=0.01)(l2)
#    l2 = Dropout(0.5)(l2)
#
#    l2 = Dense(256,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros',
#              kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),bias_regularizer=regularizers.l2(1e-5),
#              activity_regularizer=regularizers.l2(1e-6))(l2)
#    l2 = LeakyReLU(alpha=0.01)(l2)
#    l2 = Dropout(0.5)(l2)
#    pred2 = Dense(5, activation='tanh')(l2) 
#    
#    l0 = tf.add(pred1,pred2)
#    pred = tf.multiply(l0,0.5)
    pred = Dense(5, activation='softmax')(l1) 
    model_c = Model(inputs=[input0,input1,input2,input3,input4,input5],outputs=pred)
    model_c.compile(loss=categorical_crossentropy_modified,
                    optimizer=Adam(lr=1e-4),
                    metrics=['categorical_accuracy'])
    #categorical_crossentropy_modified        "categorical_crossentropy"
#    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
# =============================================================================
    Data_S00 = Data_S0.copy()
    Data_S10 = Data_S1.copy()
    Data_S20 = Data_S2.copy()
    Data_S30 = Data_S3.copy()
    Data_S40 = Data_S4.copy()
    Data_S50 = Data_S5.copy()
    label_0 = label_.copy()
    for i in range(cal_[test_index]):
        test0 += [Data_S00.pop(ti)]
        test1 += [Data_S10.pop(ti)]
        test2 += [Data_S20.pop(ti)]
        test3 += [Data_S30.pop(ti)]
        test4 += [Data_S40.pop(ti)]
        test5 += [Data_S50.pop(ti)]
        y_test += [label_0.pop(ti)]
# =============================================================================
    #Data_S00 = Data_S0[-68:] 
    #Data_S10 = Data_S1[-68:]  
    #Data_S20 = Data_S2[-68:] 
    #Data_S30 = Data_S3[-68:] 
    #Data_S40 = Data_S4[-68:] 
    #Data_S50 = Data_S5[-68:] 
    #label_0  =  label_[-68:] 
    #test0 = Data_S0[:-68]
    #test1 = Data_S1[:-68]
    #test2 = Data_S2[:-68]
    #test3 = Data_S3[:-68]
    #test4 = Data_S4[:-68]
    #test5 = Data_S5[:-68]
    #y_test = label_[:-68]
    test = [pad_sequences(test0,maxlen=seq_len,truncating='post') , pad_sequences(test1,maxlen=seq_len,truncating='post') ,
            pad_sequences(test2,maxlen=seq_len,truncating='post'), pad_sequences(test3,maxlen=seq_len,truncating='post'),
            pad_sequences(test4,maxlen=seq_len,truncating='post'), pad_sequences(test5,maxlen=seq_len,truncating='post')]    
    train = [pad_sequences(Data_S00,maxlen=seq_len,truncating='post') ,pad_sequences(Data_S10,maxlen=seq_len,truncating='post'),
             pad_sequences(Data_S20,maxlen=seq_len,truncating='post'), pad_sequences(Data_S30,maxlen=seq_len,truncating='post'),
             pad_sequences(Data_S40,maxlen=seq_len,truncating='post'), pad_sequences(Data_S50,maxlen=seq_len,truncating='post')]
# =============================================================================
#    print('normalize features ...')
#    for i in range(6):    
#        mu , sigma = Normalized(train[i])
#        train[i] = (train[i]-mu)/sigma
#        test[i] = (test[i]-mu)/sigma
#    print('Done ...')
#    ,
#              kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5),bias_regularizer=regularizers.l2(1e-5),
#              activity_regularizer=regularizers.l2(1e-6)
# =============================================================================
    y_train = np.array(label_0.copy())
    print('preparing training labels ...')
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_cat= to_categorical(y_train,5)
    print('preparing testing labels ...')
    encodert = LabelEncoder()
    encodert.fit(y_test)
    encoded_Yt = encodert.transform(y_test)
    y_catt= to_categorical(y_test,5)
    class_weights = compute_class_weight('balanced',np.unique(y_train),y_train)
    print('Training process '+str(test_index+1)+' ... of '+str(len(cal_)))
    history = model_c.fit(train,y_cat,validation_data=(test, y_catt),batch_size=128,epochs = 250)
#    model_c = load_model('/home/maouayeb/analyse-ME-DL/model/LSTM_models/lstm_models_R/LSTM_5C_'+str(test_index)+'.h5',custom_objects={'categorical_crossentropy_modified': categorical_crossentropy_modified})
#    model_c.load_weights('/home/maouayeb/analyse-ME-DL/model/LSTM_models/lstm_weights_R/LSTMWW_5C_'+str(test_index)+'.h5')
    predict_ = model_c.predict(test)
    predict = [np.argmax(i) for i in predict_]
    y_true = [np.argmax(i) for i in y_catt]
    print(confusion_matrix(y_true,predict, labels=[0,1,2,3,4]))
    cm += confusion_matrix(y_true,predict, labels=[0,1,2,3,4])
    print(cm)
    cm_ +=[confusion_matrix(y_true,predict, labels=[0,1,2,3,4])]
    loss , acc = model_c.evaluate(test,y_catt,batch_size=64)
    accuracy += acc
    print(accuracy)
    accc += [acc]
    ti += cal_[test_index]
    test0 =[]  
    test1 =[]  
    test2 =[]  
    test3 =[]  
    test4 =[]  
    test5 =[]  
    y_test = []
#    print('save LSTM model ...')
#    model_c.save('/home/maouayeb/analyse-ME-DL/model/LSTM_models/lstm_models_R/LSTM_5C_CDE(SAMM_CASMEii)_'+str(test_index)+'.h5')
#    model_c.save_weights('/home/maouayeb/analyse-ME-DL/model/LSTM_models/lstm_weights_R/LSTMWW_5C_CDE(SAMM_CASMEii)_'+str(test_index)+'.h5')
#    print('evaluate model ...')
#    model_cc = load_model('/home/maouayeb/analyse-ME-DL/LSTM_MEGC_'+str(test_index)+'.h5',custom_objects={'categorical_crossentropy_modified': categorical_crossentropy_modified})
#    model_cc.load_weights('/home/maouayeb/analyse-ME-DL/LSTMWW_MEGC_'+str(test_index)+'.h5')
    
#    del model_c
# =============================================================================
# cm = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
# for c in cm_[26:]:
#     cm += c
# accuracy=0
# for i in accc[26:]:
#     accuracy += i
# print(accuracy/len(accc[26:]))
# =============================================================================
#print(accc)  

recall0 = cm[0][0]/(cm[0][1]+cm[0][2]+cm[0][3]+cm[0][4]+cm[0][0])
#print(recall0)
recall1 = cm[1][1]/(cm[1][1]+cm[1][2]+cm[1][0]+cm[1][3]+cm[1][4])
#print(recall1)
recall2 = cm[2][2]/(cm[2][1]+cm[2][2]+cm[2][0]+cm[2][3]+cm[2][4])
#print(recall2)
recall3 = cm[3][3]/(cm[3][1]+cm[3][2]+cm[3][0]+cm[3][3]+cm[3][4])
#print(recall3)
recall4 = cm[4][4]/(cm[4][1]+cm[4][2]+cm[4][0]+cm[4][3]+cm[4][4])
#print(recall4)
test_index
preci0 = cm[0][0]/(cm[1][0]+cm[2][0]+cm[3][0]+cm[4][0]+cm[0][0])
#print(preci0)
preci1 = cm[1][1]/(cm[1][1]+cm[2][1]+cm[0][1]+cm[3][1]+cm[4][1])
#print(preci1)
preci2 = cm[2][2]/(cm[1][2]+cm[2][2]+cm[0][2]+cm[3][2]+cm[4][2])
#print(preci2)
preci3 = cm[3][3]/(cm[1][3]+cm[2][3]+cm[0][3]+cm[3][3]+cm[4][3])
#print(preci3)
preci4 = cm[4][4]/(cm[1][4]+cm[2][4]+cm[0][4]+cm[3][4]+cm[4][4])
#print(preci4)

f1_score0 = 2*(recall0*preci0)/(recall0+preci0)
f1_score1 = 2*(recall1*preci1)/(recall1+preci1)
f1_score2 = 2*(recall2*preci2)/(recall2+preci2)
f1_score3 = 2*(recall3*preci3)/(recall3+preci3)
f1_score4 = 2*(recall4*preci4)/(recall4+preci4)



fig, ax = plot_confusion_matrix(conf_mat=cm,show_normed=True)
plt.show()

# =============================================================================
# #
# ti = 0      
# w =0
# y_test = []
# print(len(label_))
# for test_index in range(len(cal_)):
#     label_0 = label_.copy()
#     for i in range(cal_[test_index]):
#         y_test += [label_0.pop(ti)]
#     
#     print('preparing testing labels ...'+str(test_index))
#     encodert = LabelEncoder()
#     encodert.fit(y_test)
#     encoded_Yt = encodert.transform(y_test)
#     y_catt= to_categorical(y_test,5)
# 
#     y_true = [np.argmax(i) for i in y_catt]
#     print(y_true)
#     w += len(y_true)
#     ti += cal_[test_index]
#     y_test = []
# 
# =============================================================================
# TP0 = 0 
# TP1 = 0 
# TP2 = 0
# TP3 = 0
# TP4 = 0 

# FP0 = 0 
# FP1 = 0 
# FP2 = 0
# FP3 = 0
# FP4 = 0 

# FN0 = 0 
# FN1 = 0 
# FN2 = 0 
# FN3 = 0
# FN4 = 0

# for j in range(len(cal_)):
#     TP0 += cm_[j][0][0]
#     TP1 += cm_[j][1][1]
#     TP2 += cm_[j][2][2]
#     TP3 += cm_[j][3][3]train
#     TP4 += cm_[j][4][4]
    
#     FP0 += cm_[j][0][1] + cm_[j][0][2] + cm_[j][0][3] + cm_[j][0][4] 
#     FP1 += cm_[j][1][0] + cm_[j][1][2] + cm_[j][1][3] + cm_[j][1][4]
#     FP2 += cm_[j][2][0] + cm_[j][2][1] + cm_[j][2][3] + cm_[j][2][4]
#     FP3 += cm_[j][3][0] + cm_[j][3][1] + cm_[j][3][2] + cm_[j][3][4]
#     FP4 += cm_[j][4][0] + cm_[j][4][1] + cm_[j][4][3] + cm_[j][4][2]
    
#     FN0 += cm_[j][1][0] + cm_[j][2][0] + cm_[j][3][0] + cm_[j][4][0]
#     FN1 += cm_[j][0][1] + cm_[j][2][1] + cm_[j][3][1] + cm_[j][4][1]
#     FN2 += cm_[j][0][2] + cm_[j][1][2] + cm_[j][3][2] + cm_[j][4][2]
#     FN3 += cm_[j][0][3] + cm_[j][1][3] + cm_[j][2][3] + cm_[j][4][3]
#     FN4 += cm_[j][0][4] + cm_[j][1][4] + cm_[j][2][4] + cm_[j][3][4]

# F1_0 = (2*TP0)/((2*TP0)+FP0+FN0)
# F1_1 = (2*TP1)/((2*TP1)+FP1+FN1)
# F1_2 = (2*TP2)/((2*TP2)+FP2+FN2)
# F1_3 = (2*TP3)/((2*TP3)+FP3+FN3)
# F1_4 = (2*TP4)/((2*TP4)+FP4+FN4)
freq = cm.sum(axis=1)
tot = freq.sum(axis=0)
print('accuracy = ', accuracy/len(cal_)) 
print('f1_score = ', (freq[0]*f1_score0+freq[1]*f1_score1+freq[2]*f1_score2+freq[3]*f1_score3+freq[4]*f1_score4)/tot)
print('UAR = ',(recall0+recall1+recall2+recall3+recall4)/5)
print('UF1 = ', (f1_score0+f1_score1+f1_score2+f1_score3+f1_score4)/5)
# print((F1_0+F1_1+F1_2+F1_3+F1_4)/5)  

#
#cmm = np.array([[54,0,4,3,0],[0,30,0,2,0],[2,0,99,2,0],[0,1,1,25,0],[4,0,6,0,15]])
#recall0 = cmm[0][0]/(cmm[0][1]+cmm[0][2]+cmm[0][3]+cmm[0][4]+cmm[0][0])
##print(recall0)
#recall1 = cmm[1][1]/(cmm[1][1]+cmm[1][2]+cmm[1][0]+cmm[1][3]+cmm[1][4])
##print(recall1)
#recall2 = cmm[2][2]/(cmm[2][1]+cmm[2][2]+cmm[2][0]+cmm[2][3]+cmm[2][4])
##print(recall2)
#recall3 = cmm[3][3]/(cmm[3][1]+cmm[3][2]+cmm[3][0]+cmm[3][3]+cmm[3][4])
##print(recall3)
#recall4 = cmm[4][4]/(cmm[4][1]+cmm[4][2]+cmm[4][0]+cmm[4][3]+cmm[4][4])
##print(recall4)
#print('UAR = ',(recall0+recall1+recall2+recall3+recall4)/5)
#
#preci0 = cmm[0][0]/(cmm[1][0]+cmm[2][0]+cmm[3][0]+cmm[4][0]+cmm[0][0])
##print(preci0)
#preci1 = cmm[1][1]/(cmm[1][1]+cmm[2][1]+cmm[0][1]+cmm[3][1]+cmm[4][1])
##print(preci1)
#preci2 = cmm[2][2]/(cmm[1][2]+cmm[2][2]+cmm[0][2]+cmm[3][2]+cmm[4][2])
##print(preci2)
#preci3 = cmm[3][3]/(cmm[1][3]+cmm[2][3]+cmm[0][3]+cmm[3][3]+cmm[4][3])
##print(preci3)
#preci4 = cmm[4][4]/(cmm[1][4]+cmm[2][4]+cmm[0][4]+cmm[3][4]+cmm[4][4])
##print(preci4)
#
#f1_score0 = 2*(recall0*preci0)/(recall0+preci0)
#f1_score1 = 2*(recall1*preci1)/(recall1+preci1)
#f1_score2 = 2*(recall2*preci2)/(recall2+preci2)
#f1_score3 = 2*(recall3*preci3)/(recall3+preci3)
#f1_score4 = 2*(recall4*preci4)/(recall4+preci4)
#
#print('f1_score = ', (f1_score0+f1_score1+f1_score2+f1_score3+f1_score4)/5)
