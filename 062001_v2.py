import cv2
import numpy as np
import mediapipe as mp
import time
import socket

# ==================== Network connection parameters ====================
UDP_PORT = 8889
TCP_PORT = 8888
DISCOVERY_MESSAGE = "DISCOVER_ARDUINO"
# This is used for identifying the target hand in the network
# Todo: Remove it from program, this piece should be working in Unity
TARGET_HAND_IDENTIFIER = "RIGHT_HAND"

# ==================== Visual Parameters ====================
# Keep it unchanged
PIP_SCALE = 0.25

# ==================== Haptic Feedback Algorithm Parameters ====================
# Based on the requirements in doc
NUM_JOINTS = 21                 # Number of joints in a hand
# Radius of the touch circle and maximum penetration depth (normalized value as a percentage of screen width)
# These values will be converted to pixel values based on the actual resolution of the camera at the start of the program
# All the code, related to devices control and collision detection should be moved to Unity
R_TOUCH_NORMALIZED = 0.2
MAX_PENETRATING_DISTANCE_NORMALIZED = 0.02
# Range of vibration amplitude and frequency
MIN_AMP = 30.0                  # Minimum vibration amplitude (0-255)
MAX_AMP = 60.0                  # Maximum vibration amplitude (0-255)
FIRST_CONTACT_FREQ = 20.0       # Vibration frequency upon first contact (Hz)
STEADY_FREQ = 0.0               # Vibration frequency during steady contact (Hz)
# Thresholds and intervals for sending commands
AMP_CHANGE_THRESHOLD = 10.0     # Only send new command if amplitude change exceeds this value
CHECK_INTERVAL = 0.1            # (seconds) Minimum interval for checking and sending haptic updates
# Not quite sure where we use it, but we'll figure it out later
ALPHA = 0.1

# ==================== 网络设置与函数 ====================
arduino_sockets = {} # 存储已连接的TCP socket

def discover_and_connect_arduinos():
    """
    使用UDP广播发现网络上的Arduino，并建立TCP连接
    """
    global arduino_sockets
    print("正在通过UDP广播发现Arduino...")
    # 创建UDP socket用于广播
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_socket.settimeout(5.0) # 设置5秒超时

    # 向局域网广播发现消息
    udp_socket.sendto(DISCOVERY_MESSAGE.encode('utf-8'), ('255.255.255.255', UDP_PORT))
    discovered_ips = {}
    try:
        while True:
            data, addr = udp_socket.recvfrom(1024)
            response = data.decode('utf-8')
            print(f"收到来自 {addr[0]} 的响应: {response}")
            # 解析响应: "HAND_IDENTIFIER IP_ADDRESS"
            parts = response.split(' ')
            if len(parts) == 2:
                hand_id, ip_addr = parts
                if hand_id == TARGET_HAND_IDENTIFIER:
                    discovered_ips[hand_id] = ip_addr
                    print(f"已发现目标设备 {TARGET_HAND_IDENTIFIER} 在 IP: {ip_addr}")
                    # 发现目标后即可停止搜索
                    break
    except socket.timeout:
        print("UDP发现超时。")

    udp_socket.close()

    if not discovered_ips:
        print(f"错误: 未能发现标识为 '{TARGET_HAND_IDENTIFIER}' 的Arduino设备。请检查Arduino是否已连接WiFi。")
        return False

    # 与发现的Arduino建立TCP连接
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
        # Arduino代码接收的joint_id是从1开始的
        lraID = joint_id + 1
        # 格式化指令: "lraID frequency amplitude\n"
        command = f"{lraID} {frequency:.2f} {amplitude:.2f}\n"
        try:
            sock.sendall(command.encode('utf-8'))
            # print(f"Sent to {hand_id} (Joint {joint_id}): {command.strip()}") # 用于调试
        except socket.error as e:
            print(f"发送指令到 {hand_id} 失败: {e}")
            # 可以尝试在这里处理重连
            arduino_sockets.pop(hand_id)

# ==================== 主程序 ====================
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
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6)

    # ---------- 新增：初始化每个关节点的触觉状态 ----------
    joint_states = []
    for _ in range(NUM_JOINTS):
        joint_states.append({
            'in_contact': False,
            'filtered_overlap': 0.0,
            'last_sent_amp': 0.0,
            'time_since_last_tx': 0.0,
        })

    # ---------- 工具函数 ----------
    def landmarks_to_np(landmarks, w, h):
        return np.array([[lm.x*w, lm.y*h] for lm in landmarks.landmark], dtype=np.float32)

    def draw_skeleton(img, pts, color=(0, 255, 0)):
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
            (13,17),(0,17),(17,18),(18,19),(19,20)]
        for i,j in connections:
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), color, 2)
        for p in pts:
            cv2.circle(img, tuple(p.astype(int)), 4, color, -1)

    # ---------- 预计算窗口大小与触觉参数（像素单位） ----------
    ret_r, tmp = cap_remote.read()
    if not ret_r:
        print("无法读取远端视频。")
        return
    h_r, w_r = tmp.shape[:2]
    pip_w, pip_h = int(w_r * PIP_SCALE), int(h_r * PIP_SCALE)

    # 将归一化的触觉参数转换为像素值
    # 我们使用远端视频的宽度作为参考基准
    r_touch_px = R_TOUCH_NORMALIZED * w_r
    max_penetrating_distance_px = MAX_PENETRATING_DISTANCE_NORMALIZED * w_r
    # 两个圆心距离小于此值即视为碰撞
    collision_threshold_px = 2 * r_touch_px
    print(f"触觉参数 (像素): 半径={r_touch_px:.2f}, 碰撞阈值={collision_threshold_px:.2f}, 最大穿透={max_penetrating_distance_px:.2f}")


    # ---------- 主循环 ----------
    last_update_time = time.time()
    while True:
        # --- 计算帧间隔时间 (dt) ---
        current_time = time.time()
        dt = current_time - last_update_time
        last_update_time = current_time

        ret_l, frame_l = cap_local.read()
        ret_r, frame_r = cap_remote.read()
        if not ret_l or not ret_r:
            break

        h_l, w_l = frame_l.shape[:2]
        pts2d_l, pts2d_r, mapped_l = None, None, None

        # --- 手部检测 (本地和远端) ---
        res_l = hands.process(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB))
        if res_l.multi_hand_landmarks:
            pts2d_l = landmarks_to_np(res_l.multi_hand_landmarks[0], w_l, h_l)
        
        res_r = hands.process(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
        if res_r.multi_hand_landmarks:
            pts2d_r = landmarks_to_np(res_r.multi_hand_landmarks[0], w_r, h_r)

        # --- 坐标映射与骨骼绘制 ---
        if pts2d_r is not None:
            draw_skeleton(frame_r, pts2d_r, color=(0, 0, 255))
        
        if pts2d_l is not None:
            pts_l = pts2d_l.copy()
            # 翻转本地手坐标并映射到远端画面空间
            pts_l[:, 0] = w_l - pts_l[:, 0]
            pts_l[:, 0] = pts_l[:, 0] / w_l * w_r
            pts_l[:, 1] = pts_l[:, 1] / h_l * h_r
            mapped_l = pts_l
            draw_skeleton(frame_r, mapped_l, color=(0, 255, 0))

        # --- 基于每个关节点距离的精细化触觉反馈 (新逻辑) ---
        if mapped_l is not None and pts2d_r is not None:
            for joint_id in range(NUM_JOINTS):
                state = joint_states[joint_id]

                # 1. 计算两用户对应关节点之间的距离和重叠量 (像素)
                distance = np.linalg.norm(mapped_l[joint_id] - pts2d_r[joint_id])
                overlap = max(0.0, collision_threshold_px - distance)

                # 2. 关节点状态机
                # 状态一: 首次接触 (之前未接触，现在重叠)
                if overlap > 0 and not state['in_contact']:
                    state['in_contact'] = True
                    state['filtered_overlap'] = 0.0  # 重置滤波器
                    amplitude = MAX_AMP              # 发送一次性最大强度的脉冲
                    frequency = FIRST_CONTACT_FREQ
                    send_haptic_command(TARGET_HAND_IDENTIFIER, joint_id, frequency, amplitude)
                    
                    state['last_sent_amp'] = amplitude
                    state['time_since_last_tx'] = 0.0 # 重置发送计时器

                # 状态二: 持续接触 (之前已接触，现在仍在重叠)
                elif overlap > 0 and state['in_contact']:
                    # 使用低通滤波器平滑重叠值，减少抖动影响
                    state['filtered_overlap'] = ALPHA * overlap + (1 - ALPHA) * state['filtered_overlap']

                    # 将平滑后的重叠值映射到振动幅度
                    penetration_ratio = state['filtered_overlap'] / max_penetrating_distance_px
                    amplitude = MIN_AMP + penetration_ratio * (MAX_AMP - MIN_AMP)
                    amplitude = np.clip(amplitude, MIN_AMP, MAX_AMP) # 保证幅度在设定范围内
                    
                    frequency = STEADY_FREQ # 频率保持不变

                    # 节流发送: 满足时间间隔且幅度变化足够大时才发送
                    state['time_since_last_tx'] += dt
                    if (state['time_since_last_tx'] >= CHECK_INTERVAL and
                        abs(amplitude - state['last_sent_amp']) >= AMP_CHANGE_THRESHOLD):
                        
                        send_haptic_command(TARGET_HAND_IDENTIFIER, joint_id, frequency, amplitude)
                        state['last_sent_amp'] = amplitude
                        state['time_since_last_tx'] = 0.0

                # 状态三: 接触结束 (之前接触，现在不重叠)
                elif overlap == 0 and state['in_contact']:
                    state['in_contact'] = False
                    state['filtered_overlap'] = 0.0
                    amplitude = 0.0
                    frequency = 0.0
                    send_haptic_command(TARGET_HAND_IDENTIFIER, joint_id, frequency, amplitude) # 发送停止指令
                    
                    state['last_sent_amp'] = 0.0
                    state['time_since_last_tx'] = 0.0
                
                # 状态四: 持续未接触 - 无需操作



        # --- 画中画 ---
        pip = cv2.flip(frame_l, 1)
        pip = cv2.resize(pip, (pip_w, pip_h))
        y1, x1 = h_r - pip_h - 10, w_r - pip_w - 10
        frame_r[y1:y1+pip_h, x1:x1+pip_w] = pip

        # --- 显示 ---
        cv2.imshow("Video Call", frame_r)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- 释放资源 ----------
    print("正在关闭程序...")
    # 发送最终停止指令给所有LRA，确保程序退出时振动停止
    print("发送最终停止指令...")
    for i in range(NUM_JOINTS):
        send_haptic_command(TARGET_HAND_IDENTIFIER, i, 0.0, 0.0)
    time.sleep(0.1) # 等待指令发送

    # 关闭sockets
    for sock in arduino_sockets.values():
        sock.close()
    cap_local.release()
    cap_remote.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 首先，连接到Arduino
    if discover_and_connect_arduinos():
        # 如果连接成功，启动主程序
        main()
    else:
        print("无法启动主程序，因为未能连接到Arduino。")