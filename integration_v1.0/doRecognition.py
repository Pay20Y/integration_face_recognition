# -*- coding: UTF-8 -*-
import face_embedding
import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser(description='face recognition')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/george/project/InsightFace/insightface/models/model-r34-amf/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--target', default=None, help='face to recognition')
args = parser.parse_args()

if args.target == None:
	print('usage: python doRecognition.py --target face_pic_path')
	exit()

model = face_embedding.FaceModel(args)

img = cv2.imread(args.target)

embedding = model.get_feature(img)

if embedding is None:
	print('error: no face detected!')
	exit()

lib_feature = np.load('lib_feature.npy')
lib_face = ['Ahmet_Needet_Sezer', 'Aicha_El_Ouafi', 'Akbar_Hashemi_Rafsanjani', 'Alberto_Fujimori', 'Bud_Selig', 'Carl_Reiner', 'Chang_Dae_wham', 'Daniel_Day_Lewis', 'Franko_Simatovic', 'Franz_Muentefering']
sim_array = np.dot(lib_feature, embedding.T)

max_sim = sim_array.max()
max_sim_index = sim_array.argmax()
if(max_sim > 0.2):
	print('success! name: ', lib_face[max_sim_index])
else:
	print('在库中无匹配人脸')
# print(sim_array)
