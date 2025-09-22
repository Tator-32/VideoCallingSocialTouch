import zmq
import socket
import time
import cv2
import imagezmq
import traceback
import simplejpeg
import threading
import queue

# ==================== Config ====================
# 假设你这台机器作为发送端绑定在 5556
# 对方机器的 Sender 绑定在 5555
LOCAL_IP = '192.168.86.226'
LOCAL_BIND_PORT = 5556
PEER_IP = '192.168.86.225'        # 对方机器 IP
PEER_SENDER_PORT = 5555        # 对方 Sender 端口

CAM_INDEX = 0
SEND_SIZE = (640, 360)         # 传输分辨率
JPEG_QUALITY = 80              # 压缩质量 80 足够清晰且带宽更低

# ==================== Exit ====================
exit_event = threading.Event()

# ==================== Queues (仅保留最新一帧) ====================
local_frame_queue  = queue.Queue(maxsize=1)
remote_frame_queue = queue.Queue(maxsize=1)

# ==================== ZMQ Objects ====================
# Sender = PUB = bind 在本机
sender = imagezmq.ImageSender(connect_to=f'tcp://{LOCAL_IP}:{LOCAL_BIND_PORT}', REQ_REP=False)
# Hub = SUB = connect 到对方
image_hub = imagezmq.ImageHub(open_port=f'tcp://{PEER_IP}:{PEER_SENDER_PORT}', REQ_REP=False)

# 关键: 只保留最新消息 防止积帧
try:
    sender.zmq_socket.setsockopt(zmq.CONFLATE, 1)
    image_hub.zmq_socket.setsockopt(zmq.CONFLATE, 1)
except Exception:
    pass
# 进一步收紧缓冲
try:
    sender.zmq_socket.setsockopt(zmq.SNDHWM, 1)
    image_hub.zmq_socket.setsockopt(zmq.RCVHWM, 1)
except Exception:
    pass

# 可选 接收超时 更快丢弃旧帧
try:
    image_hub.zmq_socket.RCVTIMEO = 200
except Exception:
    pass

rpi_name = socket.gethostname()

# ==================== Utils ====================
def q_put_latest(q: queue.Queue, item):
    if q.full():
        try:
            q.get_nowait()
        except Exception:
            pass
    q.put_nowait(item)

# ==================== Sender Thread ====================
def send_loop():
    # 建议在 macOS 保持默认后端即可
    cap = cv2.VideoCapture(CAM_INDEX)
    time.sleep(1.0)  # 给订阅端预热 防止慢加入丢首帧

    while not exit_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                continue
            if SEND_SIZE:
                frame = cv2.resize(frame, SEND_SIZE)

            now = time.time()
            msg = f"{rpi_name}*{now}"

            # 编码为 JPEG
            jpg_buffer = simplejpeg.encode_jpeg(frame, quality=JPEG_QUALITY, colorspace='BGR')

            # 发送
            sender.send_jpg(msg, jpg_buffer)

            # 本地显示只保留最新帧
            q_put_latest(local_frame_queue, frame)

        except Exception:
            print(traceback.format_exc())
            exit_event.set()
            break

    cap.release()

# ==================== Receiver Thread ====================
def recv_loop():
    while not exit_event.is_set():
        try:
            name, jpg_buffer = image_hub.recv_jpg()
            img = simplejpeg.decode_jpeg(jpg_buffer, colorspace='BGR')
            q_put_latest(remote_frame_queue, img)
        except zmq.error.Again:
            continue
        except Exception:
            print(traceback.format_exc())
            exit_event.set()
            break

# ==================== Main ====================
if __name__ == "__main__":
    print(f"[PUB] binding on *:{LOCAL_BIND_PORT}")
    print(f"[SUB] connecting to {PEER_IP}:{PEER_SENDER_PORT}")

    sender_thread = threading.Thread(target=send_loop, daemon=True)
    receiver_thread = threading.Thread(target=recv_loop, daemon=True)
    sender_thread.start()
    receiver_thread.start()

    try:
        while not exit_event.is_set():
            if not local_frame_queue.empty():
                cv2.imshow("Local", local_frame_queue.get_nowait())
            if not remote_frame_queue.empty():
                cv2.imshow("Remote", remote_frame_queue.get_nowait())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_event.set()
            # 不再额外 sleep 以减少显示端延迟
    except KeyboardInterrupt:
        exit_event.set()

    sender_thread.join()
    receiver_thread.join()
    cv2.destroyAllWindows()
    print("All threads stopped. Exiting.")
