import face_embedding
import argparse
import cv2
import numpy as np
import os

def belonging(row_index):
	if row_index in range(8):
		return 'G'
	elif row_index in range(8,16):
		return 'X'
	elif row_index in range(16,24):
		return 'W'
	else:
		return -1

image_path = "/home/george/data/DataSet1/"
# no_face_path = "no_face_image/"
not_target_path = "/home/george/data/face_detection_result/no_target_images/"
target_path = "/home/george/data/face_detection_result/target_images/"

image_size = '112,112'
model = '/home/george/project/Face_detect_recognition_all/InsightFace/insightface/models/model-r34-amf/model,0' # model to extract feature
args = (image_size, model)

model = face_embedding.FaceModel(args)
font = cv2.FONT_HERSHEY_SIMPLEX

# Feature of the target people
# lib_features = []
lib_feature_guo = np.load('lib_feature_guo.npy')
lib_feature_xi = np.load('lib_feature_xi.npy')
lib_feature_wang = np.load('lib_feature_wang.npy')
# lib_features.append(lib_feature_guo)
# lib_features.append(lib_feature_xi)
# lib_features.append(lib_feature_wang)
lib_feature = np.vstack((lib_feature_guo, lib_feature_xi))
lib_feature = np.vstack((lib_feature, lib_feature_wang))

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

	embeddings = np.array(embeddings)

	# sim_matrix_guo = np.dot(lib_features[0], embeddings.T)
	# sim_matrix_xi = np.dot(lib_features[1], embeddings.T)
	# sim_matrix_wang = np.dot(lib_features[2], embeddings.T)

	# index_x1, index_y1 = np.where(sim_matrix_guo > 0.3) # get the similarity larger than 0.3 face
	# index_x2, index_y2 = np.where(sim_matrix_xi > 0.3)
	# index_x3, index_y3 = np.where(sim_matrix_wang > 0.3)

	sim_matrix = np.dot(lib_feature, embeddings.T) # [24 * n], n is len(embeddings)

	result_face = []
	result_face_location = []
	for ie in range(sim_matrix.shape[1]):
		col_max = sim_matrix[:,ie].max()
		row_index = sim_matrix[:,ie].argmax()
		if col_max > 0.3:
			result_face.append(belonging(row_index))
			result_face_location.append(col_max)



	# Uncomment to use voting strategy
	'''
	result_face = []
	detect_face = np.unique(index_y)
	for face_id in detect_face:
		times = np.where(index_y == face_id).size
		if times > 2:
			result_face.append(face_id)
	'''
	# result_face1 = np.unique(index_y1)
	# result_face2 = np.unique(index_y2)
	# result_face3 = np.unique(index_y3)


	img_clone = img.copy()

	# if result_face1.size == 0 or result_face2.size == 0 or result_face3.size:
	if len(result_face):
		cv2.imwrite(os.path.join(not_target_path,file), img_clone)
		print('No target detected!')
	else:
		print('%d faces detected, rectangle and save...' % len(result_face))
		for i in result_face_location:
			# write max similarity on pic
			# max_sim_col = sim_matrix[:,i].max()
			cv2.rectangle(img_clone, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])), (0,0,0))
			cv2.putText(img_clone, str(result_face[i]), (int(bboxes[i][0]), int(bboxes[i][0])-3), font, 1, (0,0,0), 7)
		cv2.imwrite(os.path.join(target_path,file), img_clone)
	

		
