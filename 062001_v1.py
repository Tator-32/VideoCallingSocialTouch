import cv2
import numpy as np
import mediapipe as mp
import time
import socket

# ==================== 网络参数 ====================
UDP_PORT = 8889
TCP_PORT = 8888
DISCOVERY_MESSAGE = "DISCOVER_ARDUINO"
TARGET_HAND_IDENTIFIER = "RIGHT_HAND"  # 必须和Arduino端匹配

# ==================== 视觉与触觉参数 ====================
PIP_SCALE = 0.25

# ==================== 触觉算法常数 ====================
r_touch = 0.2                   # 每个关节点的“触摸半径”
maxPenetratingDistance = 0.02    # 最深重叠深度
minAmp = 30.0                    # 最小振幅
maxAmp = 60.0                    # 最大振幅
firstContactFreq = 20.0          # 初次接触的频率
steadyFreq = 0.0                 # 持续接触的频率
ampChangeThreshold = 10.0        # 振幅变化阈值，低于此不重发
checkInterval = 0.1              # 发送节奏（秒）
alpha = 0.1                      # 低通滤波系数

# ==================== 网络设置与函数 ====================
arduino_sockets = {}  # 存储已连接的TCP socket

def discover_and_connect_arduinos():
    """
    使用UDP广播发现网络上的Arduino，并建立TCP连接
    """
    global arduino_sockets
    print("正在通过UDP广播发现Arduino...")
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_socket.settimeout(5.0)
    udp_socket.sendto(DISCOVERY_MESSAGE.encode('utf-8'), ('255.255.255.255', UDP_PORT))
    discovered_ips = {}
    try:
        while True:
            data, addr = udp_socket.recvfrom(1024)
            response = data.decode('utf-8')
            parts = response.split(' ')
            if len(parts) == 2:
                hand_id, ip_addr = parts
                if hand_id == TARGET_HAND_IDENTIFIER:
                    discovered_ips[hand_id] = ip_addr
                    break
    except socket.timeout:
        print("UDP发现超时。")
    udp_socket.close()

    if not discovered_ips:
        print(f"错误: 未能发现标识为 '{TARGET_HAND_IDENTIFIER}' 的Arduino设备。")
        return False

    for hand_id, ip in discovered_ips.items():
        try:
            print(f"正在连接到 {hand_id} at {ip}:{TCP_PORT}...")
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_socket.connect((ip, TCP_PORT))
            arduino_sockets[hand_id] = tcp_socket
            print(f"成功连接到 {hand_id}!")
        except Exception as e:
            print(f"连接到 {hand_id} 失败: {e}")
            return False
    return True

def send_haptic_command(hand_id, joint_id, frequency, amplitude):
    """
    向指定的Arduino发送触觉指令
    """
    if hand_id in arduino_sockets:
        sock = arduino_sockets[hand_id]
        lraID = joint_id + 1  # Arduino上joint_id从1开始
        command = f"{lraID} {frequency:.2f} {amplitude:.2f}\n"
        try:
            sock.sendall(command.encode('utf-8'))
        except socket.error as e:
            print(f"发送指令到 {hand_id} 失败: {e}")
            arduino_sockets.pop(hand_id, None)

def main():
    # ---------- 初始化摄像头 ----------
    cap_local = cv2.VideoCapture(0)
    cap_remote = cv2.VideoCapture(1)
    if not cap_local.isOpened() or not cap_remote.isOpened():
        print("错误: 无法打开摄像头，请检查索引。")
        return

    # ---------- 初始化MediaPipe ----------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2)

    # ---------- 预计算窗口大小 ----------
    ret_r, tmp = cap_remote.read()
    if not ret_r:
        print("无法读取远端视频。")
        return
    h_r, w_r = tmp.shape[:2]
    pip_w, pip_h = int(w_r * PIP_SCALE), int(h_r * PIP_SCALE)

    # ---------- 初始化 per-joint 状态 ----------
    num_joints = 21
    in_contact = [False] * num_joints
    filtered_overlap = [0.0] * num_joints
    last_sent_amp = [0.0] * num_joints
    time_since_last_tx = [0.0] * num_joints

    prev_time = time.time()

    # ---------- 工具函数 ----------
    def landmarks_to_np(landmarks, w, h):
        return np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark], dtype=np.float32)

    def draw_skeleton(img, pts, color=(0, 255, 0)):
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(0,17),(17,18),(18,19),(19,20)]
        for i,j in connections:
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), color, 2)
        for p in pts:
            cv2.circle(img, tuple(p.astype(int)), 4, color, -1)

    def get_bounding_box(points):
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        return np.array([np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)])

    def check_bbox_intersection(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return xA < xB and yA < yB

    # ---------- 主循环 ----------
    while True:
        # 计算帧间时间
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        ret_l, frame_l = cap_local.read()
        ret_r, frame_r = cap_remote.read()
        if not ret_l or not ret_r:
            break

        h_l, w_l = frame_l.shape[:2]
        pts2d_l = pts2d_r = mapped_l = None

        # --- 手部检测 ---
        res_l = hands.process(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB))
        if res_l.multi_hand_landmarks:
            pts2d_l = landmarks_to_np(res_l.multi_hand_landmarks[0], w_l, h_l)

        res_r = hands.process(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
        if res_r.multi_hand_landmarks:
            pts2d_r = landmarks_to_np(res_r.multi_hand_landmarks[0], w_r, h_r)

        # --- 绘制骨骼并映射本地手到远端视图 ---
        if pts2d_r is not None:
            draw_skeleton(frame_r, pts2d_r, color=(0, 0, 255))
        if pts2d_l is not None:
            pts_l = pts2d_l.copy()
            pts_l[:, 0] = w_l - pts_l[:, 0]
            pts_l[:, 0] = pts_l[:, 0] / w_l * w_r
            pts_l[:, 1] = pts_l[:, 1] / h_l * h_r
            mapped_l = pts_l
        ##    draw_skeleton(frame_r, mapped_l, color=(0, 255, 0))

        # --- 画中画 ---
        pip = cv2.flip(frame_l, 1)
        pip = cv2.resize(pip, (pip_w, pip_h))
        y1, x1 = h_r - pip_h - 10, w_r - pip_w - 10
        frame_r[y1:y1+pip_h, x1:x1+pip_w] = pip

        # --- per-joint 触觉算法 ---
        if mapped_l is not None and pts2d_r is not None:
            for i in range(num_joints):
                # 归一化坐标到 [0,1]
                u_l, v_l = mapped_l[i][0] / w_r, mapped_l[i][1] / h_r
                u_r, v_r = pts2d_r[i][0] / w_r, pts2d_r[i][1] / h_r
                dx, dy = u_l - u_r, v_l - v_r
                distance = np.sqrt(dx*dx + dy*dy)

                # 计算重叠深度
                overlap = max(0.0, 2 * r_touch - distance)

                # 状态机
                if overlap > 0 and not in_contact[i]:
                    # 首次接触
                    in_contact[i] = True
                    filtered_overlap[i] = 0.0
                    amplitude = maxAmp
                    frequency = firstContactFreq
                    send_haptic_command(TARGET_HAND_IDENTIFIER, i, frequency, amplitude)
                    last_sent_amp[i] = amplitude
                    time_since_last_tx[i] = 0.0

                elif overlap > 0 and in_contact[i]:
                    # 持续接触，低通滤波
                    filtered_overlap[i] = alpha * overlap + (1 - alpha) * filtered_overlap[i]
                    # 线性映射到振幅范围
                    amplitude = minAmp + (filtered_overlap[i] / maxPenetratingDistance) * (maxAmp - minAmp)
                    amplitude = max(minAmp, min(amplitude, maxAmp))
                    frequency = steadyFreq
                    time_since_last_tx[i] += dt
                    # 节流发送
                    if time_since_last_tx[i] >= checkInterval and abs(amplitude - last_sent_amp[i]) >= ampChangeThreshold:
                        send_haptic_command(TARGET_HAND_IDENTIFIER, i, frequency, amplitude)
                        last_sent_amp[i] = amplitude
                        time_since_last_tx[i] = 0.0

                elif overlap == 0 and in_contact[i]:
                    # 结束接触
                    in_contact[i] = False
                    send_haptic_command(TARGET_HAND_IDENTIFIER, i, 0.0, 0.0)
                    last_sent_amp[i] = 0.0
                    time_since_last_tx[i] = 0.0
        else:
            # 当任何一只手未被检测到时
            for i in range(num_joints):
                # 检查这个关节是否之前处于接触状态
                if in_contact[i]:
                    print(f"手部丢失，停止关节 {i} 的振动。")
                    in_contact[i] = False
                    send_haptic_command(TARGET_HAND_IDENTIFIER, i, 0.0, 0.0)
                    last_sent_amp[i] = 0.0
                    time_since_last_tx[i] = 0.0
        

        # --- 显示 ---
        cv2.imshow("Video Call", frame_r)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- 释放资源 ----------
    print("正在关闭程序...")
    # 退出前确保停止所有振动
    for i in range(num_joints):
        send_haptic_command(TARGET_HAND_IDENTIFIER, i, 0.0, 0.0)
    time.sleep(0.1)

    for sock in arduino_sockets.values():
        sock.close()
    cap_local.release()
    cap_remote.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if discover_and_connect_arduinos():
        main()
    else:
        print("无法启动主程序：未能连接到Arduino。")
