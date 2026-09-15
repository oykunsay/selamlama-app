import os
import sys
import time
from collections import deque
import cv2
import numpy as np
from PIL import Image

import mediapipe as mp
import mediapipe.python.solutions.hands
import mediapipe.python.solutions.drawing_utils

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)


def get_working_camera():
    # Windows'ta varsayılan backend bazı kameralarda bozuk/boş frame
    # döndürüp "cv::Mat::Mat" assertion hatasına yol açabiliyor.
    # DirectShow (CAP_DSHOW) backend'i Windows'ta çok daha stabil.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]

    for backend in backends:
        for index in [1, 0, 2]:
            temp_cap = cv2.VideoCapture(index, backend)
            if temp_cap.isOpened():
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    return temp_cap
                temp_cap.release()
    return None


cap = get_working_camera()
if cap is None:
    print("Hata: Kamera bulunamadı!")
    exit()

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

OVERLAY_W = 300
OVERLAY_H = 533
gif_frames = []
gif_path = resource_path("selamlama.gif")

if os.path.exists(gif_path):
    gif = Image.open(gif_path)
    for frame_index in range(getattr(gif, "n_frames", 1)):
        gif.seek(frame_index)
        frame_rgba = gif.convert("RGBA")
        frame_rgba = frame_rgba.resize(
            (OVERLAY_W, OVERLAY_H), Image.Resampling.LANCZOS
        )
        gif_frames.append(np.array(frame_rgba))
    print(f"Animasyon yüklendi: {len(gif_frames)} kare.")
else:
    print("Uyarı: selamlama.gif bulunamadı!")


def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg_w), min(y + h, bg_h)

    if x1 >= x2 or y1 >= y2:
        return background

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    overlay_rgb = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    background[y1:y2, x1:x2] = (
        1.0 - alpha
    ) * background[y1:y2, x1:x2] + alpha * overlay_bgr
    return background


x_history = deque(maxlen=16)
last_wave_time = 0
COOLDOWN_SECONDS = 4
ANIMATION_DURATION = 3.5
gif_start_time = 0
wave_counter = 0


def check_wave(history):
    if len(history) < 10:
        return False

    direction_changes = 0
    significant_moves = 0

    for i in range(1, len(history) - 1):
        diff1 = history[i] - history[i - 1]
        diff2 = history[i + 1] - history[i]

        if abs(diff1) > 0.02:
            significant_moves += 1

        if (diff1 * diff2 < 0) and (abs(diff1) > 0.02 or abs(diff2) > 0.02):
            direction_changes += 1

    return direction_changes >= 2 and significant_moves >= 5


PADDING = 10
CARD_TOTAL_W = OVERLAY_W + (2 * PADDING)
MARGIN_RIGHT = 45
COMMON_X = FRAME_WIDTH - CARD_TOTAL_W - MARGIN_RIGHT

CARD_Y = 40
CARD_TOTAL_H = OVERLAY_H + (2 * PADDING)
BOX_H = 50
BOX_Y = CARD_Y + CARD_TOTAL_H + 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        # Bozuk/boş bir kare geldi, bu kareyi atla ve devam et
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()
    animation_active = (current_time - last_wave_time) < ANIMATION_DURATION

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            middle_tip_y = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ].y
            middle_tip_x = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ].x

            if middle_tip_y < wrist_y:
                x_history.append(middle_tip_x)
                if (
                    current_time - last_wave_time > COOLDOWN_SECONDS
                ) and check_wave(x_history):
                    last_wave_time = current_time
                    gif_start_time = current_time
                    wave_counter += 1
                    x_history.clear()
            else:
                x_history.clear()
    else:
        x_history.clear()

    if animation_active and gif_frames:
        elapsed = current_time - gif_start_time
        frame_idx = int(elapsed * 15) % len(gif_frames)

        cv2.rectangle(
            frame,
            (COMMON_X, CARD_Y),
            (COMMON_X + CARD_TOTAL_W, CARD_Y + CARD_TOTAL_H),
            (30, 30, 30),
            -1,
        )

        frame = overlay_transparent(
            frame,
            gif_frames[frame_idx],
            COMMON_X + PADDING,
            CARD_Y + PADDING,
        )

    # Sayaç arka planı ve mavi çerçeve
    if BOX_Y + BOX_H <= FRAME_HEIGHT and COMMON_X + CARD_TOTAL_W <= FRAME_WIDTH:
        badge_roi = frame[
            BOX_Y : BOX_Y + BOX_H, COMMON_X : COMMON_X + CARD_TOTAL_W
        ]
        badge_bg = np.zeros_like(badge_roi, dtype=np.uint8)
        badge_bg[:] = (80, 30, 15)  # Koyu mavi/lacivert arka plan
        cv2.addWeighted(badge_roi, 0.35, badge_bg, 0.65, 0, badge_roi)

        cv2.rectangle(
            frame,
            (COMMON_X, BOX_Y),
            (COMMON_X + CARD_TOTAL_W, BOX_Y + BOX_H),
            (255, 120, 0),  # Canlı Mavi / Cyan çerçeve
            2,
        )

        counter_text = f"Selamlasma Sayaci: {wave_counter}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.70
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(
            counter_text, font, font_scale, thickness
        )

        text_x = COMMON_X + (CARD_TOTAL_W - text_w) // 2
        text_y = BOX_Y + (BOX_H + text_h) // 2

        cv2.putText(
            frame,
            counter_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    cv2.imshow("Robotekno Selamlama", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('x'), ord('X')]:
        wave_counter = 0

cap.release()
cv2.destroyAllWindows()
