import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(0)

def get_custom_gesture(landmarks):
    # Define custom gestures based on landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

    if thumb_tip < index_tip and middle_tip > index_tip:
        return "volume_up"
    elif thumb_tip > index_tip and middle_tip < index_tip:
        return "volume_down"
    elif thumb_tip < index_tip and middle_tip < index_tip:
        return "mute"
    else:
        return "none"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # Draw hand landmarks and gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_custom_gesture(hand_landmarks.landmark)

            # Draw feedback text and perform actions based on gesture
            if gesture == "volume_up":
                cv2.putText(frame, "Volume Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                pyautogui.press("volumeup")
            elif gesture == "volume_down":
                cv2.putText(frame, "Volume Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                pyautogui.press("volumedown")
            elif gesture == "mute":
                cv2.putText(frame, "Mute", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                pyautogui.press("volumemute")

    # Display the resulting frame
    cv2.imshow('Hand Gesture Volume Control', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
