import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2

from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
global element
element = ''

cap = cv2.VideoCapture(0)

sys.path.append("..")


import label_map_util
import visualization_utils as vis_util

#In[0] 텐서플로우 obd api
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'


DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90




opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar_file = tarfile.open(MODEL_FILE)


for file in tar_file.getmembers():

    file_name = os.path.basename(file.name)

    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())



detection_graph = tf.Graph()


with detection_graph.as_default():

    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(os.path.join('./models/research/object_detection/data/', 'mscoco_label_map.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)\

category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)



PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]


IMAGE_SIZE = (12, 8)

class CreateUser(Resource):
    def post(self):

        return {'status': element}

app.run(debug=True)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      
      image_np_expanded = np.expand_dims(image_np, axis=0)

      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')



#In[1] visualization.py -   
      (boxes, scores, classes, num_detections) = sess.run( [boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

    
      vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=2)
      #print(vis_util.image_summary_or_default_string('image_tensor',image_np))  
      #print( np.squeeze(classes).astype(np.int32))
      #print ([classes.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))

      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

        break


