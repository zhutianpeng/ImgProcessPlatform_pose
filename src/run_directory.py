
import argparse
import logging
import time
import glob
import ast
import os
# import dill
import redis

from RedisQueue import RedisQueue
import dealimg

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

# 为了引入hashset
logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='../images/')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # 从redis里面读取
    # q_face = RedisQueue('face_forward_list')  #接收队列
    q_pose = RedisQueue('pose_forward_list')  #接收队列

    q2 = RedisQueue('test_return_list')   #发送队列
    DB_set = RedisQueue('return_set')        #缓存的set

    i = 0
    while True:
        if q_pose.empty() == False:

            #1.  接收string,转成 dict
            dic_receive = eval(q_pose.get())
            id_receive = dic_receive['id']      #接收的ID
            task_receive = dic_receive['task']
            jpg = dic_receive['img']            #接收到的是二进制的图片格式
            print("接收set测试："+ "id_receive:"+str(id_receive) + " ;task_receive: "+str(task_receive))      #测试->通过

            imgweb = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)    # 转成mat格式
            # print(imgweb)

            # image = dealimg.image_decode_openpose(base64_data, i)

            #2.  open pose deals imgs:
            # humans = e.inference(imgweb, scales=scales)
            # image = TfPoseEstimator.draw_humans(imgweb, humans, imgcopy=False)  #mat
            humans = e.inference(imgweb, scales=scales)
            resultDic = TfPoseEstimator.draw_humans_json(imgweb, humans, imgcopy=False)  # dic
            print(str(resultDic))

            # cv2.imwrite('../outimg/ %s.jpg ' % (i), image)
            # print("现在处理的图片：%s" % i)
            # i = i + 1

            # 将 mat 转成 二进制，存进 hashset 里面
            # imghtml = cv2.imencode('.jpg', image)[1].tostring() #二进制格式

            # 找到的task;img（原图）;处理结果；存成 Dic形式，转成String
            valueDic = {'task': task_receive, 'img': jpg,'result_pose':str(resultDic)}
            string_send_list = str(valueDic)


            # q2.put(imghtml)

            DB_set.setSet("imgset", id_receive, string_send_list)