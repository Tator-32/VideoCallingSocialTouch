import cv2
import imagezmq
import traceback
import time
import simplejpeg

image_hub = imagezmq.ImageHub(open_port='tcp://100.64.171.206:5555',REQ_REP=False)
frame_count = 1
time1 = 0
while True:
    try:
        time1 = time.time() if frame_count == 1 else time1
        name, image = image_hub.recv_jpg()
        # Decode
        image = simplejpeg.decode_jpeg(image, colorspace='BGR')
        cv2.imshow(name.split('*')[0],image)
        cv2.waitKey(1)
        time2 = time.time()
        # print(image.shape[:2], int(frame_count/(time2-time1)))
        frame_count += 1
    except:
        print(traceback.format_exc())
        break