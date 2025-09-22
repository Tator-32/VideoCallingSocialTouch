import socket
import time
import cv2
import imagezmq
import traceback
import simplejpeg

# Get video frame from camera
capture = cv2.VideoCapture(0)
# Fill in the ip address of host
sender = imagezmq.ImageSender(connect_to='tcp://100.64.171.205:5555', REQ_REP=False)
rpi_name = socket.gethostname()
time.sleep(2.0)
jpeg_quality = 95
while True:
    try:
        ref, frame=capture.read(0)
        time.sleep(1/60)
        image = cv2.resize(frame, (1280, 720))
        curtime = time.time
        msg = rpi_name + '*' + str(curtime)
        # simplejpeg will encode image into jpeg format
        jpg_buffer = simplejpeg.encode_jpeg(image, quality=jpeg_quality,colorspace='BGR')
        sender.send_jpg(msg, jpg_buffer)
        cv2.imshow(rpi_name, image)
        cv2.waitKey(1)
    except:
        print(traceback.print_exc)
        break