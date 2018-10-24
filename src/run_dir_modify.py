
import argparse
#import logging
import time
import glob
import ast
import os
# import six.moves.urllib as urllib
# import sys
# import tarfile
# import tensorflow as tf
# import zipfile
# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
# from PIL import Image
# import dill
from RedisQueue import RedisQueue
import dealimg

import common
import cv2
import numpy as np

#from estimator import TfPoseEstimator
# from networks import get_graph_path, model_wh

# from lifting.prob_model import Prob3dPose
# from lifting.draw import plot_pose

# from utils import label_map_util
#
# from utils import visualization_utils as vis_util
  
# logger = logging.getLogger('TfPoseEstimator')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)


# parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
# parser.add_argument('--folder', type=str, default='../images/')
# parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
# parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
# parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
# args = parser.parse_args()
# scales = ast.literal_eval(args.scales)

# w, h = model_wh(args.resolution)
# e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

# MODEL_NAME = 'myTFdata/VOC2012/VOC2012_Graph'
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_LABELS = os.path.join('myTFdata/VOC2012/data', 'pascal_label_map.pbtxt')
# NUM_CLASSES = 20

# # Load a (frozen) Tensorflow model into memory.
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')

# # Loading label map
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)

# 从redis里面读取
q1 = RedisQueue('test_forward_list_modify')  #接收队列
q2 = RedisQueue('test_return_list_modify')   #发送队列

i = 0
# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
while True:
    if q1.empty() == False:
        jpg = q1.get() #二进制
        imgweb = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)    # ma
        print(imgweb)
        # image = dealimg.image_decode_openpose(base64_data, i
        #  open pose deals imgs:
        # humans = e.inference(imgweb, scales=scales)
        # image = TfPoseEstimator.draw_humans(imgweb, humans, imgcopy=False)  #mat


        # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # imgweb_expanded = np.expand_dims(imgweb, axis=0)
        # # Definite input and output Tensors for detection_graph
        # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # # Each box represents a part of the image where a particular object was detected.
        # detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # # Each score represent how level of confidence for each of the objects.
        # # Score is shown on the result image, together with the class label.
        # detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # # Actual detection.
        # (boxes, scores, classes, num) = sess.run(
        #   [detection_boxes, detection_scores, detection_classes, num_detections],
        #   feed_dict={image_tensor: imgweb_expanded})
        # # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #   imgweb,
        #   np.squeeze(boxes),
        #   np.squeeze(classes).astype(np.int32),
        #   np.squeeze(scores),
        #   category_index,
        #   use_normalized_coordinates=True,
        #   line_thickness=8)
        # cv2.imwrite('../outimg/ %s.jpg ' % (i), image)
        print("现在处理的图片：%s" % i)
        i = i + 1
        # 将 mat 转成 二进制，存进list2里面
        imghtml = cv2.imencode('.jpg', imgweb)[1].tostring()
        q2.put(imghtml)

        # files_grabbed = glob.glob(os.path.join(args.folder, '*.jpg'))
        # # all_humans = dict()
        #
        # for i, file in enumerate(files_grabbed):
        #     # estimate human poses from a single image !
        #     image = common.read_imgfile(file, None, None)
        #     t = time.time()
        #     humans = e.inference(image, scales=scales)
        #     elapsed = time.time() - t
        #
        #     logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))
        #
        #     image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #     # cv2.imshow('tf-pose-estimation result', image)
        #     # cv2.waitKey(5)
        #
        #     filename = file.split('/')[2].split('.')[0]
        #     logger.info('image name : %s' % (filename))
        #
        #     cv2.imwrite('../outimg/ %s.jpg '%(filename), image)

        # with open(os.path.join(args.folder, 'pose.dil'), 'wb') as f:
        #     dill.dump(all_humans, f, protocol=dill.HIGHEST_PROTOCOL)
