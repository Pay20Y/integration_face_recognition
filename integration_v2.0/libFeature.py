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
args = parser.parse_args()

model = face_embedding.FaceModel(args)

libList = open('libList.txt')

num_face = 10

init_line = np.zeros(512, dtype='double')

for line in libList.readlines():
	if num_face > 0:
		face_path = os.path.join('Mr.Xi', line[:-1])
		print(face_path)
		img = cv2.imread(face_path)
		embedding_feature = model.get_feature(img)
		if embedding_feature is not None:
			num_face = num_face - 1
			init_line = np.vstack((init_line, embedding_feature))
	else:
		break
lib_feature = np.delete(init_line, 0, axis=0)
print('lib_feature size is: ', lib_feature.shape)
print(lib_feature)

np.save('lib_feature_xi.npy', lib_feature)
