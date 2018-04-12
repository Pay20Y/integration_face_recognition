from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
# import face_image
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
  def __init__(self, args):
    self.args = args
    model = edict()

    # self.threshold = 1.24
    # self.det_minsize = 50
    # self.det_threshold = [0.4,0.6,0.6]
    # self.det_factor = 0.9
    _vec = args[0].split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.image_size = image_size
    _vec = args[1].split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    # ctx = mx.gpu(args.gpu)
    ctx = mx.cpu(0)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mxnet-mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=4, accurate_landmark=False)
    self.detector = detector


  def get_feature(self, face_img):
    print ('enter get_feature')
    #face_img is bgr image
    # ret = self.detector.detect_face_limited(face_img, det_type = self.args.det)
    ret = self.detector.detect_face(face_img)
    print('detect face is done!')
    if ret is None:
      print('ret is none!')
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      print ('bbox.shape is 0!')
      return None
    num_face = bbox.shape[0]
    print('there is %d faces in the image' %num_face)
    # zhiQiao to get all faces in img
    embeddings = []
    for i in range(bbox.shape[0]):
      _bbox = bbox[i,0:4]
      _points = points[i,:].reshape((2,5)).T
      #print(bbox)
      #print(points)
      nimg = face_preprocess.preprocess(face_img, _bbox, _points, image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      # cv2.imshow('temp', nimg)
      # cv2.waitKey(-1)
      aligned = np.transpose(nimg, (2,0,1))
      #print(nimg.shape)
      embedding = None
      '''
      # print ('embedding init!')
      for flipid in [0,1]:
        if flipid==1:
          if self.args.flip==0:
            break
          do_flip(aligned)
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        _embedding = self.model.get_outputs()[0].asnumpy()
        #print(_embedding.shape)
        if embedding is None:
          embedding = _embedding
        else:
          embedding += _embedding
      # print ('embedding size is: ', embedding.shape)
      embedding = sklearn.preprocessing.normalize(embedding).flatten()
      # print ('embedding is: ', embedding)
      return embedding
      '''
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
      # print ('embedding size is: ', embedding.shape)
      embedding = sklearn.preprocessing.normalize(embedding).flatten()
      embeddings.append(embedding)
      # print ('embedding is: ', embedding)
    print('total embedding is %d' % len(embeddings))
    return embeddings, bbox
