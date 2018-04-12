import face_embedding
import argparse
import cv2
import numpy as np
import os

image_path = "/home/george/data/DataSet1/"
# no_face_path = "no_face_image/"
not_target_path = "/home/george/data/face_detection_result/no_target_images/"
target_path = "/home/george/data/face_detection_result/target_images/"

image_size = '112,112'
model = '/home/george/project/InsightFace/insightface/models/model-r34-amf/model,0' # model to extract feature
args = (image_size, model)

model = face_embedding.FaceModel(args)

for file in os.listdir(image_path):
	print('Now process %s ========>' %os.path.join(image_path,file))
	img = cv2.imread(os.path.join(image_path,file))
	# embeddings, bboxes = model.get_feature(img)
	detect_result = model.get_feature(img)

	# if embeddings is None or bboxes is None:
	if detect_result is None:
		print('There is no face!')
		continue

	embeddings = detect_result[0]
	bboxes = detect_result[1]

	lib_feature = np.load('lib_feature.npy')
	embeddings = np.array(embeddings)
	sim_matrix = np.dot(lib_feature, embeddings.T)

	index_x, index_y = np.where(sim_matrix > 0.2) # get the similarity larger than 0.2 face

	# here can add a voting strategy
	result_face = np.unique(index_y)

	img_clone = img.copy()

	if result_face.size == 0:
		cv2.imwrite(os.path.join(not_target_path,file), img_clone)
		print('No target detected!')
	else:
		print('%d faces detected, rectangle and save...' % result_face.size)
		for i in result_face:
			cv2.rectangle(img_clone, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (0,0,0))
		cv2.imwrite(os.path.join(target_path,file), img_clone)
	

		
