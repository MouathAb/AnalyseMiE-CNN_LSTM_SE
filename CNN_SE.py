#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 09:14:07 2020

@author: amouath
"""

from tensorflow.python.keras.models import load_model, Model, Input
from tensorflow.python.keras.initializers import RandomNormal
from sklearn.model_selection import train_test_split as split
#from reizeimage import resizeimage as resim
import os
import csv
import numpy as np
import pandas as pd
#from PIL import Image
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.ops import math_ops, nn, clip_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
#from tensorflow.python.keras.utils.vis_utils import plot_model
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import concatenate, Conv2D ,AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D , Activation, Flatten
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import regularizers

from PIL import Image
import tensorflow.compat.v2 as tf2
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from sklearn.utils.class_weight import compute_class_weight
import xlrd

tf.enable_eager_execution()


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
        
def flatten_vec(v):
    d = np.array([])
    for i in range(len(v)):
        d = np.append(d,v[i])
    return list(d)
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.layers import Layer

def GET_Label(File_name):
    b=[] # the objective class according to AU table 
    a=[] # the offset frame ID
    c=[] # the Onset frame ID
    label=[]
    if (File_name.endswith('SAMM_Micro_FACS_Codes_v2.xlsx')):
        ss = 14 # from where starts the lalbels
        s = 10 # the coluymn of the objective class
    else:
        ss = 1 # from where starts the lalbels
        s = 9 # the coluymn of the objective class ! it doesn't exist in default 
              # version but since we have information about the AU so we can
              # create the column of the corresponded objective classes
    
    L= open_excel(File_name)
    for i in range(ss,len(L)):
        b.append(int(L[i][s])-1)
        a.append(int(L[i][5]))
        c.append(int(L[i][3]))
    
    for j in range(len(a)):
        label.append(b[j])

    return label,b,a,c
def relabel(u,Re):
    u0 =[]
    for i in range(len(u)):
        if Re in [0,1]:
            if u[i] == 1:
                u0 += [7]
            else:
                u0 += [u[i]]
        elif Re in [3,4]:
            if u[i] == 1:
                u0 += [u[i]]
            else:
                u0 += [7]
        elif Re == 2:
            if u[i] == 0:
                u0 += [u[i]]
            else:
                u0 += [7]
        else:
            u0 += [u[i]]
    return u0
encodert = LabelEncoder()

def load_data2(url,u,cn,RR):
#    images = []
    
    encoded_Yt = encodert.transform(u)          
    uu = to_categorical(encoded_Yt,cn)
    image = [] 
    #test = []
    for j,x in enumerate(url):
        image += [x[RR]]

    return image,uu
''' [(39, 30),(39, 30),(46, 31),(24, 23),(24, 23),(62, 23)] : (toul,3ardh)'''
BUFFER_SIZE = 500
BATCH_SIZE = 32

def psnet(x, beta=1.0, gama=0.0):
    return  k.sigmoid(beta * (x-gama))
class Psnet(Layer):

    def __init__(self, beta=1.0, gama=0.0, **kwargs):
        super(Psnet, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.gama = gama
        self.__name__ = 'Psnet'

    def build(self, input_shape):
#        self.beta = k.variable(self.beta,
#                                      dtype=k.floatx(),
#                                      name='beta_factor')
#        self.gama = k.variable(self.gama,
#                                      dtype=k.floatx(),
#                                      name='gama_factor')
        self.beta = self.add_weight(name='beta',
                                        shape=(1,1),
                                        initializer='uniform',
                                        trainable=True,
                                        constraint=None)
        self.gama = self.add_weight(name='gama',
                                        shape=(1,1),
                                        initializer='uniform',
                                        trainable=True,
                                        constraint=None)

        super(Psnet, self).build(input_shape)

    def call(self, inputs, mask=None):
        return psnet(inputs, self.beta, self.gama)

    def get_config(self):
        config = {'beta': self.get_weights()[0],
                  'gama': self.get_weights()[1]} 
        base_config = super(Psnet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape   
def process_image(image):
    x = IWG_WIDTH
    y = IMG_HEIGHT
    image = tf2.image.decode_png(image, channels=3)
    image = tf2.image.resize(image, (x*4,y*4))
    image /= 127.5
    image = image -1
    
    sous_region =[tf2.image.crop_to_bounding_box(image,0,0,x,y), 
                  tf2.image.crop_to_bounding_box(image,0,y,x,y),
                  tf2.image.crop_to_bounding_box(image,0,2*y,x,y),
                  tf2.image.crop_to_bounding_box(image,0,3*y,x,y),
                  tf2.image.crop_to_bounding_box(image,x,0,x,y),
                  tf2.image.crop_to_bounding_box(image,x,y,x,y),
                  tf2.image.crop_to_bounding_box(image,x,2*y,x,y),
                  tf2.image.crop_to_bounding_box(image,x,3*y,x,y),
                  tf2.image.crop_to_bounding_box(image,2*x,0,x,y),
                  tf2.image.crop_to_bounding_box(image,2*x,y,x,y),
                  tf2.image.crop_to_bounding_box(image,2*x,2*y,x,y),
                  tf2.image.crop_to_bounding_box(image,2*x,3*y,x,y),
                  tf2.image.crop_to_bounding_box(image,3*x,0,x,y),
                  tf2.image.crop_to_bounding_box(image,3*x,y,x,y),
                  tf2.image.crop_to_bounding_box(image,3*x,2*y,x,y),
                  tf2.image.crop_to_bounding_box(image,3*x,3*y,x,y),
                  ] # the 4/9/16 patches
#    image = tf2.image.resize_with_pad(image, IWG_WIDTH,IMG_HEIGHT)
#    image /= 255
#    image = math_ops.cast(image, tf2.float16)
#    image = tf.image.convert_image_dtype(image, np.uint8, saturate=True)
    
    return sous_region
def load_and_preprocess_image(path):
    image = tf2.io.read_file(path)
    return process_image(image)

def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.
  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.
  Returns:
      A tensor.
  """
  return ops.convert_to_tensor(x, dtype=dtype)
_EPSILON = K.epsilon()
def categorical_crossentropy_modified(target, output, from_logits=False, axis=-1):
  rank = len(output.shape)
  axis = axis % rank
  # Note: nn.softmax_cross_entropy_with_logits_v2
  # expects logits, Keras expects probabilities.
  if not from_logits:
    # scale preds so that the class probas of each sample sum to 1
    output = output / math_ops.reduce_sum(output, axis, True)
    # manual computation of crossentropy
    epsilon_ = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
    return -math_ops.reduce_sum(target *math_ops.squared_difference(1.0,output)* math_ops.log(output), axis)
  else:
    return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output)
def prepare_for_training(ds, cache=True, shuffle_buffer_size=400, s=True):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        if s:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
def train_all(Re,labelnnz,Data_urlz,y_train,x_train,m):
    labelnn_v2_train = y_train
    labelnn_v2_test= labelnnz
    class_num = len(np.unique(labelnn_v2_train))
    try:
        class_weights = compute_class_weight('balanced',np.unique(labelnn_v2_test),labelnn_v2_test)
    except:
        class_weights = compute_class_weight('balanced',np.unique(labelnn_v2_train),labelnn_v2_train)
        
    encodert.fit(labelnn_v2_train)
    
    encoded_Ytrain = encodert.transform(labelnn_v2_train)          
    y = to_categorical(encoded_Ytrain,class_num)
    
    encoded_Ytest = encodert.transform(labelnn_v2_test)          
    yy = to_categorical(encoded_Ytest,class_num)
    
    trainx0 = tf.data.Dataset.from_tensor_slices(x_train)
    trainx0 = trainx0.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
    train_Y = tf.data.Dataset.from_tensor_slices(y)
    train_ds = tf.data.Dataset.zip((trainx0, train_Y))
    
    testx0 = tf.data.Dataset.from_tensor_slices(Data_urlz)
    testx0 = testx0.map(load_and_preprocess_image)
    test_Y = tf.data.Dataset.from_tensor_slices(yy)
    test_ds = tf.data.Dataset.zip((testx0, test_Y))
            
    
    train_ds = prepare_for_training(train_ds, s = True)
    test_ds = prepare_for_training(test_ds, s = False)
    
    cnn =[]
    inputCNN = []

    for R in range(m):
        inputCNN += [Input(shape=(IWG_WIDTH,IMG_HEIGHT,3))]
                
#                CNN = Conv2D(16, (1, 1) , activation='relu',padding='same',kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros')(inputCNN[R])
        CNN = Conv2D(4, (5, 5) , activation='relu',padding='same',kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros')(inputCNN[R])
        MaxPool00 = MaxPooling2D(pool_size=(2,2), padding='same')(CNN)
        CNN0 = Conv2D(8, (3, 3) , activation='relu',padding='same',kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros')(MaxPool00)
        MaxPool0 = MaxPooling2D(pool_size=(2,2), padding='same')(CNN0)
        
        MaxPool00 = MaxPooling2D(pool_size=(2,2), padding='same')(MaxPool0)
        
        CNN1 = Conv2D(16, (1, 1) , activation='relu',padding='same',kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros')(MaxPool0)
        MaxPool1 = MaxPooling2D(pool_size=(2,2), padding='same')(CNN1)
        
        CNN2 = Conv2D(16, (3, 3) , activation='relu',padding='same',kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros')(MaxPool0)
        MaxPool2 = MaxPooling2D(pool_size=(2,2), padding='same')(CNN2)
        
        CNN3 = Conv2D(16, (5, 5) , activation='relu',padding='same',kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros')(MaxPool0)
        MaxPool3 = MaxPooling2D(pool_size=(2,2), padding='same')(CNN3)
        
        CNN4 = Conv2D(16, (7, 7) , activation='relu',padding='same',kernel_initializer=RandomNormal(stddev=0.1),bias_initializer='zeros')(MaxPool0)
        MaxPool4 = MaxPooling2D(pool_size=(2,2), padding='same')(CNN4)
        
        concat = concatenate([MaxPool00, MaxPool1, MaxPool2, MaxPool3, MaxPool4])

        cnn += [concat]

    conc = concatenate([cnn[_] for _ in range(m)])

    feat = int(conc.shape[3])
    alpha = int(feat/4)
    a_se = Conv2D(alpha,[1,1],padding='same',
                  activation='relu',
                  kernel_initializer=RandomNormal(stddev=0.1),
                  bias_initializer='zeros')(conc)
#           a_se = LeakyReLU(alpha=0.1)(a_se)
#a_se0 = MaxPooling2D(pool_size=(2,2), padding='same')(a_se0)
    a_se0 = Conv2D(feat,[1,1],padding='same',
                   activation = 'sigmoid',
                   kernel_initializer=RandomNormal(stddev=0.1),
                   bias_initializer='zeros')(a_se)


    a_tot = tf.multiply(conc,a_se0) #LSTM-SE : with a_tot2
    a_tot = Conv2D(162,[1,1],padding='same',
                   activation='relu',
                   kernel_initializer=RandomNormal(stddev=0.1),
                   bias_initializer='zeros')(a_tot)
    # =============================================================================
    #        a_tot = tf.multiply(a_se0,conc)
    l = GlobalAveragePooling2D()(a_tot)
            
#            l = LeakyReLU(alpha=0.1)(l)
            
    l = Dense(2048,activation='relu',kernel_initializer=RandomNormal(stddev=0.01))(l)
#            l = LeakyReLU(alpha=0.1)(l)
    l = Dropout(0.5)(l)
    l = Dense(256,activation='relu',kernel_initializer=RandomNormal(stddev=0.01))(l)
#            l = LeakyReLU(alpha=0.1)(l)
    l = Dropout(0.5)(l)
#    l = Dense(20,activation='relu',kernel_initializer=RandomNormal(stddev=0.01))(l)
##            l = LeakyReLU(alpha=0.1)(l)
#    l = Dropout(0.5)(l)
#            l = Dense(20,kernel_initializer=RandomNormal(stddev=0.01),bias_initializer='zeros')(l)
#            l = LeakyReLU(alpha=0.1)(l)
#            l = Dropout(0.5)(l)
    pred = Dense(class_num,activation="softmax")(l)
            
            
    modelf = Model(inputs=inputCNN,outputs=pred)
        #    modelf.summary()
            
        
#    callbacks = [ModelCheckpoint(str(Re)+'_best_CNN.h5',monitor='val_categorical_accuracy', save_best_only=True, save_weights_only=True)]    
    modelf.compile(loss=categorical_crossentropy_modified,
                   optimizer=Adam(lr=1e-4),
                   metrics=['categorical_accuracy'])
            #  categorical_crossentropy_modified     "categorical_crossentropy" Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
            #   ,class_weight=class_weights,callbacks=callbacks
    modelf.fit(train_ds,epochs=60,class_weight=class_weights)
#            modelf.load_weights(str(Re)+'_best_CNN.h5')
    modell = tf.keras.Model(inputs=modelf.input, outputs=modelf.layers[-3].output)
            #modell.summary()
#            labelnn_v2 = relabel(labelnn,Re)
#            datas,ys = load_data2(Data_url,labelnn_v2,class_num,Re)
#            trainx0s = tf.data.Dataset.from_tensor_slices(datas)
#            trainx0s = trainx0s.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
#            train_Ys = tf.data.Dataset.from_tensor_slices(ys)
#            train_dss = tf.data.Dataset.zip((trainx0s, train_Ys))
#            train_dss = prepare_for_training(train_dss, s = False)
#            predy = modell.predict(train_dss)
    print('extract features ...')
    del modelf, train_ds, y, yy, cnn
    del trainx0,train_Y,testx0,test_Y,inputCNN
    gc.collect()
    pred_y = modell.predict(test_ds)
#    predy = modell.predict(train_ds)
#            with open('*test_' + str(test_index)+"_region_"+str(Re)+".csv", "w") as output:
#                writer = csv.writer(output)
#                writer.writerows(predy)
#            output.close
    del modell , test_ds
    gc.collect()
    print('save features ...')
    with open('_5_clases_test_'+m+'P_region_'+str(Re)+".csv", "w") as output:
        writer = csv.writer(output)
        writer.writerows(pred_y)
    output.close
#    with open('train__region_'+str(Re)+".csv", "w") as output:
#        writer = csv.writer(output)
#        writer.writerows(predy)
#    output.close
    
    gc.collect()
    K.clear_session()
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

def fix_label (k, labeln_):
    labelnn_s = []
    for R in range(6):
        labelnn_ss = []
        for j , l  in enumerate(labeln_[R]):
            for i in range(k[j]):
                labelnn_ss += [l]
        labelnn_s += [labelnn_ss]
    leel_s = labelnn_s.copy()
    ooo_s = []
    for R in range(6):
        oo_s = []
        for i in range(len(labelnn_s[R])):
            if labelnn_s[R][i] in [5,6]:
                oo_s += [i - len(oo_s)]
        ooo_s += [oo_s]
    
    label_lii_s =[]
    for R in range(len(ooo_s)):
        label_li_s = labelnn_s[R].copy()
        for j in range(len(ooo_s[R])):
            label_li_s.pop(ooo_s[R][j])
        label_lii_s += [label_li_s]
    labelnn_s = label_lii_s.copy()
            
    for R in range(6):
        for i in range(len(labelnn_s[R])):
            if labelnn_s[R][i] == 7:
                labelnn_s[R][i]=5
    return labelnn_s,  ooo_s, leel_s

def adjust_sizes(Data_url,lel,sizes):

    Data=[]
    for R in range(6):
        Datas =[]
        for j,x in enumerate(Data_url):
            if lel[R][j] not in [5,6]:
                Datas += [x[R]]
        print(len(Datas))
        Data+=[Datas]
        
    return Data

# =============================================================================
#                            ~ start here ~ 
# =============================================================================
print("get Image url DATA ...")  
samm_url = '/mnt/DONNEES/Bureau/analyse-ME-DL/'
Data_urlx_samm = pd.read_csv(samm_url+'Data_url_SAMM.csv',header=None)
Data_url_samm = Data_urlx_samm.values.tolist()   
# correct url file 
for i in range(len(Data_url_samm)):
    for j in range(6):
        Data_url_samm[i][j] = Data_url_samm[i][j][:6]+'amouath/Bureau'+Data_url_samm[i][j][14:] 
        
casmeii_url = '/mnt/DONNEES/Bureau/analyse-ME-DL/'
Data_urlx_casmeii_ = pd.read_csv(casmeii_url+'Data_url_CASMEII.csv',header=None)
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
label_samm = "/mnt/DONNEES/Bureau/SAMM/" 
File_name_s = label_samm + "SAMM_Micro_FACS_Codes_v2.xlsx"
label_s,b_s,a_s,c_s = GET_Label(File_name_s)
labeln_s = arrange_label(label_s,File_name_s)

print("get Label's CASMEII (AU) from source ...")   
label_casmeii = '/mnt/DONNEES/download/'
File_name_c = label_casmeii+'CASME2-coding-20190701.xlsx'
label_c,b_c,a_c,c_c = GET_Label(File_name_c)
labeln_c = arrange_label(label_c,File_name_c)

# Create a "subject" list that contains a "sequence" list 

# For SAMM 
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
#For CASMEII
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

# Convert label by sequence to label by frame 
labelnn_s,  ooo_s, leel_s = fix_label(k_s, labeln_s)
labelnn_c,  ooo_c, leel_c = fix_label(k_c, labeln_c)

# Get the size of each region
sizes = []
for R in range(6):
    url_0 = '/mnt/DONNEES' + Data_url_casmeii[0][R][13:]
    f_0 = Image.open(url_0)
    f_0_w,f_0_h=f_0.size
    sizes.append([f_0_h,f_0_w])

# Adjust the size of the images 
Data_s = adjust_sizes(Data_url_samm,leel_s,sizes)
Data_c = adjust_sizes(Data_url_casmeii,leel_c,sizes)

# Mix the CASMEII and the SAMM data and labels
Data_url =  np.concatenate((Data_c,Data_s),axis=1)
labelnn = np.concatenate((labelnn_c,labelnn_s),axis=1)
AUTOTUNE = tf.data.experimental.AUTOTUNE
print('done ...')

#from natsort import natsorted, ns

#labelnn = Data_urlx.label.tolist()
# =============================================================================
# clean data from unnecessary frames 
# =============================================================================
#Data_url , labelnn = clean(Data_url,labelnn)Data_url_casmeii
# =============================================================================
#Re = 5 #Region number
# =============================================================================
ti = 0
#Epochs = [48,48,48,48,48,48] 10249;24

# =============================================================================
# Correct the url of data
# if your data is in ddifferent place you may change it here
for _ in range(6):
    for url in range(len(Data_url[_])):
        Data_url[_,url] = '/mnt/DONNEES/' + Data_url[_,url][13:]

# Split data
Data_url1 = np.transpose(Data_url)
labelnn1 = np.transpose(labelnn)
x_train,x_test,y_train,y_test = split(Data_url1,labelnn1,test_size=0.5, random_state=42)
# =============================================================================
x_train = np.transpose(x_train)
y_train = np.transpose(y_train)

#    class_weights = compute_class_weight('balanced',np.unique(labelnnz),labelnnz)

# train ~ Valid ~ save features
m = 9 #the number of patches
for Re in range(6):
    if not os.path.exists('_5_clases_test_16P_region_'+str(Re)+".csv"):
        print('#'*25);print("region : "+str(Re));print("#"*25)
        si00 = 0 
        si10 = 0
        for i in Data_url:
            size = Image.open('/mnt/DONNEES' +i[Re][13:]).size
            si00 += size[0] 
            si10 += size[1]
        si0 = int(si00/len(Data_url)) # mean width
        si1 = int(si10/len(Data_url)) # mean hight
        print(si0,si1)
        IWG_WIDTH = int(si0/2)
        IMG_HEIGHT = int(si1/2)
        train_all(Re,labelnn[Re],Data_url[Re],y_train[Re],x_train[Re],m)
        gc.collect()


''' 0 : 0.9931 / 0.9937 / 0.9987 , 1 : 0.9617 / 0.9878 / 0.9954 , 2 : 0.9729 / 0.9939 / 0.9922 ,
    3 : 0.9038 / 0.9713 / 0.9904 , 4 : 0.9130 / 0.9741 / 0.9975 , 5 : 0.9609 / 0.9859 / 0.9793'''
