import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque, Counter

pyautogui.PAUSE = 0.01 
pyautogui.FAILSAFE = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

last_fingers_count = -1
last_action_time = 0
last_volume_time = 0
prev_time = 0

# 滑动窗口大小为5
gesture_buffer = deque(maxlen=5)

hud_items = [
    (0, "zero      : Play/Pause"),
    (1, "one       : Volume -"),
    (2, "two       : Volume +"),
    (3, "three     : PPT Next"),
    (4, "four      : PPT Prev"),
    (5, "five      : New Tab"),
    (6, "six       : Close Tab"),
    (7, "seven     : Mute Volume"),
    (8, "eight     : Page Down"),
    (9, "nine      : Page Up"),
    (10, "ten      : Refresh Tab")
]

def dist_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def count_single_hand_fingers(hand_landmarks, force_count=False):
    p0 = hand_landmarks.landmark[0]   # 手腕
    p4 = hand_landmarks.landmark[4]   # 大拇指尖
    p5 = hand_landmarks.landmark[5]   # 食指根
    p6 = hand_landmarks.landmark[6]   # 食指第二关节
    p8 = hand_landmarks.landmark[8]   # 食指尖
    p9 = hand_landmarks.landmark[9]   # 中指根
    p10 = hand_landmarks.landmark[10]
    p12 = hand_landmarks.landmark[12] # 中指尖
    p14 = hand_landmarks.landmark[14]
    p16 = hand_landmarks.landmark[16] # 无名指尖
    p17 = hand_landmarks.landmark[17] # 小指根
    p18 = hand_landmarks.landmark[18]
    p20 = hand_landmarks.landmark[20] # 小指尖

    palm_width = dist_2d(p5, p17)
    if palm_width < 0.01:
        palm_width = 0.01

    # 归一化指尖距离，防止手在镜头前伸远伸近导致绝对距离失效
    dist_4_8_norm = dist_2d(p4, p8) / palm_width
    dist_4_12_norm = dist_2d(p4, p12) / palm_width
    dist_8_12_norm = dist_2d(p8, p12) / palm_width

    index_up = dist_2d(p8, p0) > dist_2d(p6, p0)
    middle_up = dist_2d(p12, p0) > dist_2d(p10, p0)
    ring_up = dist_2d(p16, p0) > dist_2d(p14, p0)
    pinky_up = dist_2d(p20, p0) > dist_2d(p18, p0)
    
    palm_direction = p5.x - p17.x
    thumb_up = False
    if palm_direction > 0:
        if p4.x > hand_landmarks.landmark[3].x:
            thumb_up = True
    else:
        if p4.x < hand_landmarks.landmark[3].x:
            thumb_up = True

    index_hooked = (dist_2d(p8, p0) < dist_2d(p6, p0)) and (dist_2d(p8, p0) > dist_2d(p5, p0))

    # 判断指尖是否完全扣死在指根
    index_folded_fist = dist_2d(p8, p5) / palm_width < 0.4
    middle_folded_fist = dist_2d(p12, p9) / palm_width < 0.4

    # 只要没有强制要求数数，就优先判定特殊手势
    if not force_count:
        # 手势6
        if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
            return "CHINESE_6"
            
        # 手势7
        is_pinched = (dist_4_8_norm < 0.5) and (dist_4_12_norm < 0.5) and (dist_8_12_norm < 0.4)
        if is_pinched and not ring_up and not pinky_up and not index_folded_fist and not middle_folded_fist:
            return "CHINESE_7"
            
        # 手势8
        if thumb_up and index_up and not index_hooked and not middle_up and not ring_up and not pinky_up:
            return "CHINESE_8"
            
        # 手势9
        if index_hooked and not middle_up and not ring_up and not pinky_up:
            return "CHINESE_9"
            
        # 手势10
        if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
            if dist_8_12_norm < 0.35:
                return "CHINESE_10"
            
    # 特殊手势未命中或被多手冲突阻断，走常规数手指逻辑
    extended = 0
    if thumb_up: extended += 1
    if index_up: extended += 1
    if middle_up: extended += 1
    if ring_up: extended += 1
    if pinky_up: extended += 1
    return extended

cap = cv2.VideoCapture(0)
print("手势控制器已启动。按 'q' 键退出。")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # 遮罩层逻辑，避免HUD字样被复杂的背景掩盖
    hud_overlay = img.copy()
    cv2.rectangle(hud_overlay, (w - 290, 10), (w - 10, 500), (30, 30, 30), -1)
    cv2.addWeighted(hud_overlay, 0.4, img, 0.6, 0, img)
    
    # 没手的时候也得初始化变量，防溢出报错
    smoothed_fingers = -1
    action_text = "IDLE"
    text_color = (150, 150, 150)
    
    if results.multi_hand_landmarks:
        total_fingers = 0
        num_detected_hands = len(results.multi_hand_landmarks)
        
        # 分流处理：单手才认特殊中文手势。双手时强制降级为纯数手指
        if num_detected_hands == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            res = count_single_hand_fingers(hand_landmarks, force_count=False)
            if res == "CHINESE_6": total_fingers = 6
            elif res == "CHINESE_7": total_fingers = 7
            elif res == "CHINESE_8": total_fingers = 8
            elif res == "CHINESE_9": total_fingers = 9
            elif res == "CHINESE_10": total_fingers = 10
            else: total_fingers = res
        else:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                total_fingers += count_single_hand_fingers(hand_landmarks, force_count=True)
            
        # 滑动窗口求众数，过滤变手势过程中的过渡帧杂音
        gesture_buffer.append(total_fingers)
        smoothed_fingers = Counter(gesture_buffer).most_common(1)[0][0]
            
        now = time.time()
        cv2.putText(img, f"TOTAL VALUE: {smoothed_fingers}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # --- 连续触发逻辑组 (音量) ---
        if smoothed_fingers in [1, 2]:
            last_fingers_count = smoothed_fingers  # 立即记录，避免阻塞后面的单次手势
            if smoothed_fingers == 1:
                action_text = "VOLUME DOWN"
                text_color = (0, 180, 255)
                if now - last_volume_time > 0.15:
                    pyautogui.press('volumedown')
                    last_volume_time = now
            elif smoothed_fingers == 2:
                action_text = "VOLUME UP"
                text_color = (0, 255, 0)
                if now - last_volume_time > 0.15:
                    pyautogui.press('volumeup')
                    last_volume_time = now
                
        # --- 连续触发逻辑组 (翻页/滚动) ---
        elif smoothed_fingers in [3, 4, 8, 9]:
            last_fingers_count = smoothed_fingers  # 同步状态机
            if smoothed_fingers == 3:
                action_text = "PPT NEXT PAGE"
                text_color = (255, 255, 0)
                if now - last_action_time > 0.8:
                    pyautogui.press('right')
                    last_action_time = now
            elif smoothed_fingers == 4:
                action_text = "PPT PREV PAGE"
                text_color = (255, 100, 0)
                if now - last_action_time > 0.8:
                    pyautogui.press('left')
                    last_action_time = now
            elif smoothed_fingers == 8:
                action_text = "PAGE DOWN"
                text_color = (0, 255, 255)
                if now - last_action_time > 0.5:
                    pyautogui.press('pagedown')
                    last_action_time = now
            elif smoothed_fingers == 9:
                action_text = "PAGE UP"
                text_color = (0, 255, 255)
                if now - last_action_time > 0.5:
                    pyautogui.press('pageup')
                    last_action_time = now

        # --- 单次触发逻辑组 ---
        elif smoothed_fingers in [0, 5, 6, 7, 10]:
            if smoothed_fingers == 0: 
                action_text = "PLAY / PAUSE"
                text_color = (255, 0, 255)
            elif smoothed_fingers == 5: 
                action_text = "NEW BROWSER TAB"
                text_color = (0, 255, 128)
            elif smoothed_fingers == 6: 
                action_text = "CLOSE BROWSER TAB"
                text_color = (255, 50, 50)
            elif smoothed_fingers == 7: 
                action_text = "MUTE VOLUME"
                text_color = (255, 128, 0)
            elif smoothed_fingers == 10: 
                action_text = "REFRESH TAB"
                text_color = (128, 0, 255)

            if smoothed_fingers != last_fingers_count:
                if now - last_action_time > 0.8:
                    if smoothed_fingers == 0:
                        pyautogui.press('space')
                    elif smoothed_fingers == 5:
                        pyautogui.hotkey('ctrl', 't') 
                    elif smoothed_fingers == 6:
                        pyautogui.hotkey('ctrl', 'w') 
                    elif smoothed_fingers == 7:
                        pyautogui.press('volumemute')
                    elif smoothed_fingers == 10:
                        pyautogui.press('f5')
                    
                    last_fingers_count = smoothed_fingers
                    last_action_time = now
    else:
        cv2.putText(img, "NO HAND DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        last_fingers_count = -1
        gesture_buffer.clear()  # 移出视野时清空，防止下次手伸进来时残留脏数据

    # 动态渲染HUD菜单
    cv2.putText(img, "DUAL-HAND & TRAD SHORTCUTS", (w - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
    for i, (val, label) in enumerate(hud_items):
        y_pos = 70 + i * 40
        if results.multi_hand_landmarks and smoothed_fingers == val:
            color = (0, 255, 0)      # 激活状态
            thickness = 2
        else:
            color = (200, 200, 200)  # 静态待命
            thickness = 1
        cv2.putText(img, f"[{val}] {label}", (w - 270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, thickness)

    # 简易FPS计数器
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(img, f"FPS: {int(fps)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(img, f"ACTION: {action_text}", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    cv2.imshow("HCI Dual-Hand Controller Pro", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
