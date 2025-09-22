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
# All the code related to device control and collision detection should be moved to Unity
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

# ==================== Network setup and functions ====================
arduino_sockets = {} # Store connected TCP sockets

def discover_and_connect_arduinos():
    """
    Use UDP broadcast to discover Arduinos on the network and establish TCP connections
    """
    global arduino_sockets
    print("Discovering Arduino via UDP broadcast...")

    # [OFFLINE] Skipped hardware discovery for testing
    discovered_ips = {}

    # [OFFLINE] Offline mode: skip hardware discovery and connection
    print("[OFFLINE] Skipping hardware discovery and connection, entering recognition test mode.")
    return True

def send_haptic_command(hand_id, joint_id, frequency, amplitude):
    """
    Send haptic command to the specified Arduino
    """
    if hand_id in arduino_sockets:
        sock = arduino_sockets[hand_id]
        # Arduino expects joint_id starting from 1
        lraID = joint_id + 1
        # Format: "lraID frequency amplitude\n"
        command = f"{lraID} {frequency:.2f} {amplitude:.2f}\n"
        try:
            sock.sendall(command.encode('utf-8'))
        except socket.error as e:
            print(f"Failed to send command to {hand_id}: {e}")
            arduino_sockets.pop(hand_id)

# ==================== Main Program ====================
def main():
    # ---------- Initialize cameras ----------
    cap_local = cv2.VideoCapture(0)
    cap_remote = cv2.VideoCapture(1)
    if not cap_local.isOpened() or not cap_remote.isOpened():
        print("Error: Unable to open cameras, please check indices.")
        return

    # ---------- Initialize MediaPipe ----------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6)

    # ---------- Initialize haptic state for each joint ----------
    joint_states = []
    for _ in range(NUM_JOINTS):
        joint_states.append({
            'in_contact': False,
            'filtered_overlap': 0.0,
            'last_sent_amp': 0.0,
            'time_since_last_tx': 0.0,
        })

    # ---------- Utility functions ----------
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

    # ---------- Precompute window size and haptic parameters (pixels) ----------
    ret_r, tmp = cap_remote.read()
    if not ret_r:
        print("Unable to read remote video.")
        return
    h_r, w_r = tmp.shape[:2]
    pip_w, pip_h = int(w_r * PIP_SCALE), int(h_r * PIP_SCALE)

    # Convert normalized haptic parameters to pixels
    r_touch_px = R_TOUCH_NORMALIZED * w_r
    max_penetrating_distance_px = MAX_PENETRATING_DISTANCE_NORMALIZED * w_r
    collision_threshold_px = 2 * r_touch_px
    print(f"Haptic params (pixels): radius={r_touch_px:.2f}, collision threshold={collision_threshold_px:.2f}, max penetration={max_penetrating_distance_px:.2f}")

    # ---------- Main loop ----------
    last_update_time = time.time()
    while True:
        current_time = time.time()
        dt = current_time - last_update_time
        last_update_time = current_time

        ret_l, frame_l = cap_local.read()
        ret_r, frame_r = cap_remote.read()
        if not ret_l or not ret_r:
            break

        h_l, w_l = frame_l.shape[:2]
        pts2d_l, pts2d_r, mapped_l = None, None, None

        # --- Hand detection (local and remote) ---
        res_l = hands.process(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB))
        if res_l.multi_hand_landmarks:
            pts2d_l = landmarks_to_np(res_l.multi_hand_landmarks[0], w_l, h_l)
        
        res_r = hands.process(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))
        if res_r.multi_hand_landmarks:
            pts2d_r = landmarks_to_np(res_r.multi_hand_landmarks[0], w_r, h_r)

        # --- Coordinate mapping and skeleton drawing ---
        if pts2d_r is not None:
            draw_skeleton(frame_r, pts2d_r, color=(0, 0, 255))
        
        if pts2d_l is not None:
            pts_l = pts2d_l.copy()
            # Mirror local hand coordinates and map to remote image space
            pts_l[:, 0] = w_l - pts_l[:, 0]
            pts_l[:, 0] = pts_l[:, 0] / w_l * w_r
            pts_l[:, 1] = pts_l[:, 1] / h_l * h_r
            mapped_l = pts_l
            draw_skeleton(frame_r, mapped_l, color=(0, 255, 0))

        # --- Haptic feedback logic based on joint distance ---
        if mapped_l is not None and pts2d_r is not None:
            for joint_id in range(NUM_JOINTS):
                state = joint_states[joint_id]

                distance = np.linalg.norm(mapped_l[joint_id] - pts2d_r[joint_id])
                overlap = max(0.0, collision_threshold_px - distance)

                # State 1: First contact
                if overlap > 0 and not state['in_contact']:
                    state['in_contact'] = True
                    state['filtered_overlap'] = 0.0
                    amplitude = MAX_AMP
                    frequency = FIRST_CONTACT_FREQ
                    # [OFFLINE] Disabled sending
                    state['last_sent_amp'] = amplitude
                    state['time_since_last_tx'] = 0.0

                # State 2: Continuous contact
                elif overlap > 0 and state['in_contact']:
                    state['filtered_overlap'] = ALPHA * overlap + (1 - ALPHA) * state['filtered_overlap']
                    penetration_ratio = state['filtered_overlap'] / max_penetrating_distance_px
                    amplitude = MIN_AMP + penetration_ratio * (MAX_AMP - MIN_AMP)
                    amplitude = np.clip(amplitude, MIN_AMP, MAX_AMP)
                    frequency = STEADY_FREQ

                    state['time_since_last_tx'] += dt
                    if (state['time_since_last_tx'] >= CHECK_INTERVAL and
                        abs(amplitude - state['last_sent_amp']) >= AMP_CHANGE_THRESHOLD):
                        # [OFFLINE] Disabled sending
                        state['last_sent_amp'] = amplitude
                        state['time_since_last_tx'] = 0.0

                # State 3: Contact ended
                elif overlap == 0 and state['in_contact']:
                    state['in_contact'] = False
                    state['filtered_overlap'] = 0.0
                    amplitude = 0.0
                    frequency = 0.0
                    # [OFFLINE] Disabled sending
                    state['last_sent_amp'] = 0.0
                    state['time_since_last_tx'] = 0.0

        # --- Picture-in-picture ---
        pip = cv2.flip(frame_l, 1)
        pip = cv2.resize(pip, (pip_w, pip_h))
        y1, x1 = h_r - pip_h - 10, w_r - pip_w - 10
        frame_r[y1:y1+pip_h, x1:x1+pip_w] = pip

        # --- Display ---
        cv2.imshow("Video Call", frame_r)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- Release resources ----------
    print("Closing program...")
    print("Sending final stop commands...")
    for i in range(NUM_JOINTS):
        # [OFFLINE] Disabled sending
        pass
    time.sleep(0.1)

    for sock in arduino_sockets.values():
        sock.close()
    cap_local.release()
    cap_remote.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if discover_and_connect_arduinos():
        main()
    else:
        print("Unable to start main program because Arduino connection failed.")
