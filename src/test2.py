
from RedisQueue import RedisQueue
import glob
import os
import dealimg

import asyncio
import websockets
import base64
import dealimg
import cv2
import numpy as np

q1 = RedisQueue('test_forward_list')  #发送队列
q2 = RedisQueue('test_return_list')   #接收队列

# windows
# files_grabbed = glob.glob(os.path.join('d:\\FileTest\\before', '*.jpg'))
# linux
# files_grabbed = glob.glob(os.path.join('../FileTest/before', '*.jpg'))
#
# for file in enumerate(files_grabbed):
#     print(file)
#     base64_data = dealimg.image_encode(file[1])
#     q.put(base64_data)



async def hello(websocket, path):
     count = 1
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
                # base64_data = base64.b64encode(jpg)
                # print("message1 %s" %base64_data)
                q1.put(jpg)

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
                if q2.empty() == False:
                    jpghtml = q2.get()
                    await websocket.send(jpghtml)

         count=count+1



start_server = websockets.serve(hello, '10.103.238.52', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()