import cv2 
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [4, 8, 12, 16, 20]

with mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_image)
        finger_states = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))

                if lm_list:
                    if lm_list[4][1] > lm_list[3][1]:
                        finger_states.append(1)
                    else: 
                        finger_states.append(0)

                for tip in finger_tips[1:]:
                    if lm_list[tip][2] < lm_list[tip - 2][2]:
                        finger_states.append(1)
                    else:
                        finger_states.append(0)

            text = f"Fingers: {finger_states}"
            cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand tracker", image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()