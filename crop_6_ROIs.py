#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:20:51 2019

@author: maouayeb
"""

from sklearn import preprocessing
import os
import numpy as np

#import pandas as pd
from PIL import Image
from imutils import face_utils
#import imutils
import dlib
#import cv2
from resizeimage import resizeimage

import glob
import xlrd

from matplotlib import pyplot as plt
import csv



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

def DownLoad_DB (Dir_name):
    a = np.array([])
    c = glob.glob(Dir_name+'/*')
    for j in c:
        b = glob.glob(j+'/*')
        for i in b:
            a=np.append(a,glob.glob(i+'/*.jpg'))
    a = sorted(a)
    return a


def get_subject_Sequence_Frame_number(imagee,Data_1):
    p=[]
    e=0
    f=0
    text=imagee.split("/")
    sub = text[8][3:]
    ima = text[10][3:len(text[10])]
    tez=""
    for d in sorted(Data_1) :
        texti=d.split("/")
        if d[:61].endswith(sub) : # 0:61 is the par of the url that ends with the subject number 
            if  texti[9] != tez :
                p.append(texti[9])
                tez = texti[9]

    for k,c in enumerate(p):
        if c.endswith(text[9]):  
            e=k
#    
#    
    s = int(sub)-1 
    f = int(ima) - 1
    return s,e,f



def face_detection (images):
    detector = dlib.get_frontal_face_detector()
    rect = detector(images, 1)
    return rect


def Landmarks(imagel,rect):
    
    P= "shape_predictor_68_face_landmarks.dat" # Landmarks from a pretrained HOG algorithm this file should be with the code file in the same directory 
    predictor = dlib.shape_predictor(P)
    for k,d in enumerate(rect):
        shape = predictor(imagel, d)
        shape = face_utils.shape_to_np(shape)
        
    return shape


def Face_align_croping(image,shape_np):
    x1 = shape_np[17][0]
    y1 = min(shape_np[18][1],shape_np[25][1])
    x2 = shape_np[26][0]
    y2 = shape_np[8][1]
    box=(x1,y1,x2,y2)
    imag = Image.fromarray(image,'RGB')
    r = imag.crop(box)
#    window= dlib.image_window()
#    window.set_image(image)    
    return r


def regions_crop(facer,SS,EE,FF):
    
#    detector = dlib.get_frontal_face_detector()
#    P= "shape_predictor_68_face_landmarks.dat" # Landmarks from a pretrained HOG algorithm this file should be with the code file in the same directory 
#    predictor = dlib.shape_predictor(P)
#    rect_r = detector(facer,1)
#    for k,d in enumerate(rect_r):
#        sp = predictor(facer,d)
#        sp = face_utils.shape_to_np(sp)
    '''
    Choice of regions of interest :necessary morphological patches (NMPs) => choice of the significant Landmarks
    6 Regions:
         As descriped in the article : An improved Micro expression Recognition Method Ba
                 Yue et al.
        
    '''

    
    # contains the bow of each region
    box=[[0,100,20,100],[140,240,20,100],[60,180,80,160],[0,60,120,180],[180,240,120,180],[40,200,180,240]]
    face_im = resizeimage.resize_contain(facer, [240, 280])
    url =[]
    for R in range(6):
        region = face_im.crop((box[R][0],box[R][2],box[R][1],box[R][3]))
        if not os.path.exists("/home/maouayeb/analyse-ME-DL/RADS/Region_"+str(R)):
            os.makedirs("/home/maouayeb/analyse-ME-DL/RADS/Region_"+str(R))
        if not os.path.exists("/home/maouayeb/analyse-ME-DL/RADS/Region_"+str(R)+"/Subject_"+str(SS)):
            os.makedirs("/home/maouayeb/analyse-ME-DL/RADS/Region_"+str(R)+"/Subject_"+str(SS))
        if not os.path.exists("/home/maouayeb/analyse-ME-DL/RADS/Region_"+str(R)+"/Subject_"+str(SS)+"/Sequence_"+str(EE)):
            os.makedirs("/home/maouayeb/analyse-ME-DL/RADS/Region_"+str(R)+"/Subject_"+str(SS)+"/Sequence_"+str(EE))
        url_0 = "/home/maouayeb/analyse-ME-DL/RADS/Region_"+str(R)+"/Subject_"+str(SS)+"/Sequence_"+str(EE)+"/frame_"+str(FF)+".png"
        region.save(url_0)
        url.append(url_0)
    return url

def GET_Label(File_name):
    b=[]
    a=[]
    label=[]
    L= open_excel(File_name)
    for i in range(len(L)-2):
        b.append(str(L[i][9]))
        a.append(int(L[i][5])-int(L[i][3])+1)
    
    for j in range(len(a)):
        d=[]
        for k in range(a[j]):
            d.append(b[j])
        label.append(d)
    return label,b,a


def Update_labels(labels):
    LE = preprocessing.LabelEncoder()
    LE.fit(['happiness','others','repression','disgust','fear','sadness','surprise','repression'])
    labels = flatten_vec(labels)
    result = LE.transform(labels)
    print(LE.classes_)
    return result
    
def flatten_im(s):
    d=[]
    for h in range(len(s)):
        for i in range(len(s[h])):
            if str(type(s[h][i]))[8].endswith('l') :
                for j in range(len(s[h][i])):    
                    d.append(s[h][i][j])
            else:
                d.append(s[h][i])
    return d 

def flatten_vec(v):
    d = np.array([])
    for i in range(len(v)):
        d = np.append(d,v[i])
    return list(d)

######################################################################################################
######################################################################################################


#Download data : data est un fichier text contient l'url de chaque image 
'''
CASME 2 Data base
'''
Dir_name ="/home/maouayeb/Bureau/data/CASME/CASME2/CASME2_RAW_selected/CASME2_RAW_selected"
Dir_name = "/home/maouayeb/Bureau/CASME/CASME2/CASME2_RAW/CASME2RAW"
Data_url_1 = DownLoad_DB(Dir_name)
'''
    Maybe we will need a preprocessing to normalize the frames with TIM (time interpolation model)
'''
    # Landmarks avec Active Shape Model + Recadrage + Alignement + Décoppage en ROIs
'''
    The following block must be executed only one times 
'''


                                                 # |!| begin |!|
     
Data_1=[]
for i in sorted(Data_url_1):
    Data_1.append(i[:len(i)-4]) 

EE =""
Data_url = []
shapes =[]
for image in sorted(Data_1):
    S,E,F= get_subject_Sequence_Frame_number(image,sorted(Data_1))
#        config = Path("/home/maouayeb/analyse-ME-DL/RADS/Region_0/Subject_"+str(S)+"/Sequence_"+str(E)+"/frame_"+str(F)+".png")
#        if not config.is_file():
    im = dlib.load_rgb_image(image+".jpg")
    if (EE != E):
        EE = E
        detector = dlib.get_frontal_face_detector()
        P= "shape_predictor_68_face_landmarks.dat" # Landmarks from a pretrained HOG algorithm this file should be with the code file in the same directory 
        predictor = dlib.shape_predictor(P)
        rect_1 = detector(im, 1)
        for k,d in enumerate(rect_1):
            shape = predictor(im, d)  
            shape_np = face_utils.shape_to_np(shape)
    face = Face_align_croping(im,shape_np) 
    nmp =regions_crop(face,S,E,F) # nmp: Necessary Morphological Patchs
    Data_url.append(nmp)
    #save Data urls    
with open('Data_url.csv','w') as csvFile:
     writer = csv.writer(csvFile)
     writer.writerows(Data_url)
csvFile.close()
    #Prepare the data : charger les labels de la ase de données puis donner à chaque régions le label adéquat
File_name = "label_mouath.xlsx"
labels,b,a = GET_Label(File_name)
labeln = Update_labels(b)
    #labelr = region_label(labeln)
    #save labels
seq = list(labeln)
with open('labeln.txt','w') as f:
    for item in seq:
        f.write("%f\n" % item) 
    
    """
    visualisation of the distribution of the labels : histogram
    """
plt.hist(b)
plt.title('histogram of labels : unbalanced data')
plt.xlabel('emotions')
plt.ylabel('count')
plt.show()

    #adjust size of each region above all the data because we need a fixed size image to train the CNN model
'''
    In this stage the variable Data_url should be a vector with lines under the below form:
        ./A_sized/Region_xx_yy/subject_nn/frame_mm
        with:
            xx: Region number
            yy: image's size of the region xx
            nn: subject number
        mm: frame or image number
The variable Data should be a vector of matrix that represent the frames ordred as mentioned in the Data_url        
'''
        
                                            # |!| end  |!|    









