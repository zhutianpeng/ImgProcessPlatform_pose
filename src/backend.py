from RedisQueue import RedisQueue
import glob
import os
import dealimg
import redis
import asyncio
import websockets
import base64
import dealimg
import cv2
import numpy as np

q_face = RedisQueue('face_forward_list')  #face发送队列
q_pose = RedisQueue('pose_forward_list')  #pose发送队列
q2 = RedisQueue('test_return_list')   #接收队列
DB_set = RedisQueue('return_set')        #缓存的set

# 1. 修改图片帧的数据，做测试数据

# 2. hashset 的数据结构的定义

# 3. 输出队列的定义

# 4. 人脸识别程序的接入


async def hello(websocket, path):
     count = 1
     id_send = 1
     id_receive = 0
     while True:
         if count==1:
            message =await websocket.recv()
            await websocket.send(message)
            print("message: %s"%message)
         else:
            message1 =await websocket.recv()  # base64

            message = base64.b64decode(message1)  # 将客户端发送过来的base64解码为二进制
            a = message.find(b'\xff\xd8' )
            b = message.find(b'\xff\xd9' )

            if a != -1 and b != -1:      # 能找到上述字符

                jpg = message[a:b+2]

                # 图片帧定义修改
                dict_send_list = {'id': id_send,'task': 1,'img':jpg}
                string_send_list = str(dict_send_list)
                # print("发送帧数据: "+string_send_list)

                # 存入队列
                # q_face.put(string_send_list)  暂时不需要
                q_pose.put(string_send_list)
                id_send = id_send+1


            # #     # 将base64转成Mat格式，暂时不需要
            #     imgweb = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            #
            #     #   在此处将Mat结果传到redis里面
            #     print(imgweb)
            #     q.put(imgweb)
            #
            #     # 人脸识别，框出人脸，暂时不需要
            #     images,img = faceutil.detect_faces(faceutil.model_face, imgweb)
            #     cv2.imshow("video", img)
            #     if cv2.waitKey(1)==27:
            #         exit(0)
            #
            #     # 将mat转成 “二进制” 格式
            #     imghtml = cv2.imencode('.jpg', img)[1].tostring()
            # #    print(imghtml)
            #
                # await websocket.send(imghtml)
                # await websocket.send(jpg)
                #          从数据库里面取图片，并且给前端
                # if q2.empty() == False:
                #     jpghtml = q2.get()
                #     await websocket.send(jpghtml)

                if DB_set.sSize("imgset") != 0 :
                    result_str = DB_set.getSet("imgset",id_receive)
                    if result_str is None:
                        pass
                    else:
                        dic_result_str = eval(result_str)
                        print("后端接受到的set的result：" + dic_result_str['result_pose']+" set的task是："+ dic_result_str['task'])
                    id_receive = id_receive + 1

         count=count+1



start_server = websockets.serve(hello, '10.103.238.125', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()