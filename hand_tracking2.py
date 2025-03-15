import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to calculate stable 3D angles
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Vectors
    ba = a - b
    bc = c - b
    
    # Normalize vectors
    ba /= np.linalg.norm(ba)
    bc /= np.linalg.norm(bc)
    
    # Compute angle (in degrees)
    cosine_angle = np.dot(ba, bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return round(angle, 2)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Check camera permissions!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            # Extract key points for fingers
            wrist = np.array([lm[0].x, lm[0].y, lm[0].z])

            # Thumb landmarks
            thumb_cmc = np.array([lm[1].x, lm[1].y, lm[1].z])
            thumb_mcp = np.array([lm[2].x, lm[2].y, lm[2].z])
            thumb_ip = np.array([lm[3].x, lm[3].y, lm[3].z])
            thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])

            # Index landmarks
            index_mcp = np.array([lm[5].x, lm[5].y, lm[5].z])
            index_pip = np.array([lm[6].x, lm[6].y, lm[6].z])
            index_dip = np.array([lm[7].x, lm[7].y, lm[7].z])
            index_tip = np.array([lm[8].x, lm[8].y, lm[8].z])

            # Middle landmarks
            middle_mcp = np.array([lm[9].x, lm[9].y, lm[9].z])
            middle_pip = np.array([lm[10].x, lm[10].y, lm[10].z])
            middle_dip = np.array([lm[11].x, lm[11].y, lm[11].z])
            middle_tip = np.array([lm[12].x, lm[12].y, lm[12].z])

            # Ring landmarks
            ring_mcp = np.array([lm[13].x, lm[13].y, lm[13].z])
            ring_pip = np.array([lm[14].x, lm[14].y, lm[14].z])
            ring_dip = np.array([lm[15].x, lm[15].y, lm[15].z])
            ring_tip = np.array([lm[16].x, lm[16].y, lm[16].z])

            # Pinky landmarks
            pinky_mcp = np.array([lm[17].x, lm[17].y, lm[17].z])
            pinky_pip = np.array([lm[18].x, lm[18].y, lm[18].z])
            pinky_dip = np.array([lm[19].x, lm[19].y, lm[19].z])
            pinky_tip = np.array([lm[20].x, lm[20].y, lm[20].z])

            # Calculate angles (0° to 180°)
            thumb_angle = calculate_angle(thumb_cmc, thumb_mcp, thumb_tip)
            index_angle = calculate_angle(index_mcp, index_pip, index_tip)
            middle_angle = calculate_angle(middle_mcp, middle_pip, middle_tip)
            ring_angle = calculate_angle(ring_mcp, ring_pip, ring_tip)
            pinky_angle = calculate_angle(pinky_mcp, pinky_pip, pinky_tip)

            # Calculate thumb sidewise movement
            thumb_side_angle = calculate_angle(thumb_mcp, wrist, index_mcp)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display angles
            cv2.putText(frame, f"Thumb: {thumb_angle}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Thumb Side: {thumb_side_angle}°", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Index: {index_angle}°", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Middle: {middle_angle}°", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Ring: {ring_angle}°", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Pinky: {pinky_angle}°", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Show webcam feed
    cv2.imshow("Hand Tracking - Finger Angles (0-180°)", frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
