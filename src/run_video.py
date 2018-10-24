import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

# e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
# humans = e.inference(image)
# image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='../etcs/person.avi')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))  #ztp: step1
    #logger.debug('cam read+')
    #cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(args.video)
    #ret_val, image = cap.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # ztp part

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # videoWriter = cv2.VideoWriter('../etcs/VideoOut/123.avi', fourcc, 25, (640,480))  # (1280,720)为视频大小

    time_begin =time.time()

    with open('../etcs/resultfile.txt', 'w') as f:
        while(cap.isOpened()):
            fps_time = time.time()
            ret_val, image = cap.read()

            try:
                humans = e.inference(image)  #ztp:step2
                f.write('\n'+'result:')
                for human in humans:
                    f.write("123humans part:%s" % human)
            except Exception:
                print("处理完成")
                break;
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False) #ztp:step3

            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

            # cv2.imshow('tf-pose-estimation result', image)
            # fps_time = time.time()
            # if cv2.waitKey(1) == 27:
            #     break

            # videoWriter.write(image)

    # videoWriter.release()
    # cv2.destroyAllWindows()
logger.debug('finished+')
print("time: %f" % (time.time() - time_begin))
