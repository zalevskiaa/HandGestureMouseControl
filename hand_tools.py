import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands


def compute_hand_middle(img, hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger_mcp = \
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    wrist_pos = (int(wrist.x * img.shape[1]), int(wrist.y * img.shape[0]))
    index_pos = (int(index_finger_mcp.x * img.shape[1]), 
                 int(index_finger_mcp.y * img.shape[0]))
    pinky_pos = (int(pinky_mcp.x * img.shape[1]), 
                 int(pinky_mcp.y * img.shape[0]))

    center_x = int((wrist_pos[0] + index_pos[0] + pinky_pos[0]) / 3)
    center_y = int((wrist_pos[1] + index_pos[1] + pinky_pos[1]) / 3)
    center_pos = (center_x, center_y)
    return center_pos


def draw_hand_middle(img, hand_landmarks, color=(0, 255, 0)):
    center_pos = compute_hand_middle(img, hand_landmarks)
    return cv2.circle(img, center_pos, 5, color, -1)


def distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 +
                   (point1.y - point2.y)**2 +
                   (point1.z - point2.z)**2)


def compute_palm_size(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger_mcp = \
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    palm_size = distance(wrist, index_finger_mcp) + distance(wrist, pinky_mcp)

    return palm_size


def is_thumb_index_touching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    palm_size = compute_palm_size(hand_landmarks)
    thumb_index_distance = distance(thumb_tip, index_tip)
    relative_distance = thumb_index_distance / palm_size

    return relative_distance < 0.15


def is_thumb_middle_touching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = \
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    palm_size = compute_palm_size(hand_landmarks)
    thumb_index_distance = distance(thumb_tip, index_tip)
    relative_distance = thumb_index_distance / palm_size

    return relative_distance < 0.15
