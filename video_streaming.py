import zmq
import socket
import time
import cv2
import imagezmq
import traceback
import simplejpeg
import threading
import queue

# ========== Exit thread ==========
exit_event = threading.Event()

# ========== Queues for frames ==========
local_frame_queue = queue.Queue()
remote_frame_queue = queue.Queue()

# ========== Sender Config ==========
sender = imagezmq.ImageSender(connect_to='tcp://192.168.86.226:5556', REQ_REP=False)
rpi_name = socket.gethostname()
jpeg_quality = 95

# ========== Receiver Config ==========
image_hub = imagezmq.ImageHub(open_port='tcp://192.168.86.225:5555', REQ_REP=False)
image_hub.zmq_socket.RCVTIMEO = 500 # Set time out range to 500ms

# ========== Sender Thread ==========
def send_loop():
    capture = cv2.VideoCapture(0)
    time.sleep(2.0)
    while not exit_event.is_set():
        try:
            ret, frame = capture.read()
            if not ret:
                continue
            image = cv2.resize(frame, (1280, 720))
            curtime = time.time()
            msg = rpi_name + '*' + str(curtime)
            jpg_buffer = simplejpeg.encode_jpeg(image, quality=jpeg_quality, colorspace='BGR')
            sender.send_jpg(msg, jpg_buffer)
            if not local_frame_queue.full():
                local_frame_queue.put_nowait(image)
        except:
            print(traceback.format_exc())
            exit_event.set()
            break
    capture.release()

# ========== Receiver Thread ==========
def recv_loop():
    while not exit_event.is_set():
        try:
            name, image = image_hub.recv_jpg()
            image = simplejpeg.decode_jpeg(image, colorspace='BGR')
            if not remote_frame_queue.full():
                remote_frame_queue.put_nowait(image)
        except zmq.error.Again:
            continue
        except:
            print(traceback.format_exc())
            exit_event.set()
            break

# ========== Main ==========

if __name__ == "__main__":
    sender_thread = threading.Thread(target=send_loop, daemon=True)
    receiver_thread = threading.Thread(target=recv_loop, daemon=True)
    sender_thread.start()
    receiver_thread.start()

    try:
        while not exit_event.is_set():
            # Local video display
            if not local_frame_queue.empty():
                frame = local_frame_queue.get_nowait()
                cv2.imshow("Local", frame)

            # Remote video display
            if not remote_frame_queue.empty():
                frame = remote_frame_queue.get_nowait()
                cv2.imshow("Remote", frame)

            # Keyboard input to control exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_event.set()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected.")
        exit_event.set()

    sender_thread.join()
    receiver_thread.join()
    cv2.destroyAllWindows()
    print("All threads stopped. Exiting.")