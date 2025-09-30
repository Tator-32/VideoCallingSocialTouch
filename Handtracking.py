import cv2
import numpy as np
import mediapipe as mp
import time
import socket

# ==================== Main Program ====================
def main():
    # ---------- Initialize cameras ----------
    cap_local = cv2.VideoCapture(2)
    if not cap_local.isOpened():
        print("Error: Unable to open cameras, please check indices.")
        return

    # ---------- Initialize MediaPipe ----------
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6)

    # ---------- Utility functions ----------
    def landmarks_to_np(landmarks, w, h):
        return np.array([[lm.x*w, lm.y*h] for lm in landmarks.landmark], dtype=np.float32)

    # Draw skeleton on image
    def draw_skeleton(image, landmarks, color=(0, 255, 0)):
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start_point = tuple(landmarks[start_idx].astype(int))
            end_point = tuple(landmarks[end_idx].astype(int))
            cv2.line(image, start_point, end_point, color, 2)
        for idx, lm in enumerate(landmarks):
            center = tuple(lm.astype(int))
            cv2.circle(image, center, 5, color, -1)

    # ---------- Main loop ----------
    last_update_time = time.time()
    while True:
        current_time = time.time()
        dt = current_time - last_update_time
        last_update_time = current_time

        ret_l, frame_l = cap_local.read()
        frame_l = cv2.flip(frame_l, 1)  # Mirror local frame for correct orientation
        if not ret_l:
            break

        h_l, w_l = frame_l.shape[:2]
        pts2d_l = None

        # --- Hand detection (local and remote) ---
        res_l = hands.process(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB))
        if res_l.multi_hand_landmarks:
            pts2d_l = landmarks_to_np(res_l.multi_hand_landmarks[0], w_l, h_l)

        
        if pts2d_l is not None:
            draw_skeleton(frame_l, pts2d_l)

        # --- Picture-in-picture ---
        pip = cv2.flip(frame_l, 1)

        # --- Display ---
        cv2.imshow("Video Call", frame_l)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- Release resources ----------
    print("Closing program...")
    print("Sending final stop commands...")
    time.sleep(0.1)
    cap_local.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
