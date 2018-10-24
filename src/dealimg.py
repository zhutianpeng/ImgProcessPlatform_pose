import base64
import cv2
import numpy as np

def convert_image():
    # Picture ==> base64 encode
    with open('d:\\FileTest\\Hope_Despair.jpg', 'rb') as fin:
        image_data = fin.read()
        base64_data = base64.b64encode(image_data)

        fout = open('d:\\FileTest\\base64_content.txt', 'w')
        fout.write(base64_data.decode())
        fout.close()

        # base64 encode ==> Picture
    with open('d:\\FileTest\\base64_content.txt', 'r') as fin:
        base64_data = fin.read()
        ori_image_data = base64.b64decode(base64_data)

        fout = open('d:\\FileTest\\Hope_Despair_2.jpg', 'wb')
        fout.write(ori_image_data)
        fout.close()

if __name__ == '__main__':
    convert_image()

def image_encode(file):
    with open(file, 'rb') as fin:
        image_data = fin.read()
        base64_data = base64.b64encode(image_data)
    return base64_data

def image_decode(data,filename):
    base64_data = data
    ori_image_data = base64.b64decode(base64_data)

    # windows
    # fout = open('d:\\FileTest\\after\\%s.jpg' %filename, 'wb')
    # linux
    fout = open('../FileTest/after/%s.jpg' % filename, 'wb')

    fout.write(ori_image_data)
    fout.close()

def image_decode_openpose(data,filename):
    base64_data = data
    ori_image_data = base64.b64decode(base64_data)

    nparr = np.fromstring(ori_image_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img_np

def opencv_to_base64(img_np):
    image = cv2.imencode('.jpg', img_np)[1]
    base64_data = str(base64.b64encode(image))[2:-1]
    return base64_data
