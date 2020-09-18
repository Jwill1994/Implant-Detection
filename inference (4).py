import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


#%matplotlib inline

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
##############ws + 
import xml.etree.ElementTree as ET
from scipy.spatial import KDTree
from statistics import mean
# What model to download.
#MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH =  'trained_inference_graphs/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'annotations/implant.pbtxt'
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'inference_test'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(0, 60)] #0,60
# Size, in inches, of the output images.
IMAGE_SIZE = (90, 30)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        
        #print(detection_boxes)
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
#ws +
def bb_intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
  # return the intersection over union value
  return iou
iou_all = []
for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  print(image_np.shape)
  sh = image_np.shape
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  #print(output_dict)
  print('---------------')
  ########ws +#########
  #print(output_dict['detection_boxes'][0])
  bb = output_dict['detection_boxes'][0]
  (xmin,ymin,xmax,ymax) = (bb[1]*sh[1], bb[0]*sh[0], bb[3]*sh[1], bb[2]*sh[0] )
  print(xmin,ymin,xmax,ymax)
  predicted_bb = (xmin,ymin,xmax,ymax)
  ####xml###############################
  tree = ET.parse(image_path.split('.')[0]+'.xml')
  root = tree.getroot()
  gt_coord = []
  for group in root.findall('object'):
    title = group.find('bndbox')
    ph1 = title.find('xmin').text
    ph2 = title.find('ymin').text
    ph3 = title.find('xmax').text
    ph4 = title.find('ymax').text
    gt_coord.append((int(ph1),int(ph2),int(ph3),int(ph4)))
  kdtree = KDTree(gt_coord)
  neighbor = kdtree.query(predicted_bb)
  neighbor_bb = gt_coord[neighbor[1]]
  ws_iou = bb_intersection_over_union(predicted_bb,neighbor_bb)
  print(ws_iou)
  iou_all.append(ws_iou)
    #print(titlephrase)
  '''
  # Visualization of the results of a detection.
  bb = output_dict['detection_boxes'][1]
  (xmin,ymin,xmax,ymax) = (bb[1]*sh[1], bb[0]*sh[0], bb[3]*sh[1], bb[2]*sh[0] )
  print(xmin,ymin,xmax,ymax)
  bb = output_dict['detection_boxes'][2]
  (xmin,ymin,xmax,ymax) = (bb[1]*sh[1], bb[0]*sh[0], bb[3]*sh[1], bb[2]*sh[0] )
  print(xmin,ymin,xmax,ymax)
  bb = output_dict['detection_boxes'][3]
  (xmin,ymin,xmax,ymax) = (bb[1]*sh[1], bb[0]*sh[0], bb[3]*sh[1], bb[2]*sh[0] )
  print(xmin,ymin,xmax,ymax)
  '''
  '''
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  
  
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
  '''
import csv
print('iou for all test images', iou_all)
with open('output_iou.csv','w',newline='') as f:
  writer = csv.writer(f)
  writer.writerow(iou_all)
print('my calculated average iou:', mean(iou_all))