import cv2
import requests
import numpy as np
import mediapipe as mp
import threading

import pyautogui
import time


from hand_tools import is_thumb_index_touching, is_thumb_middle_touching, \
                        draw_hand_middle, compute_hand_middle
from screen_tools import compute_screen_coords


class ThreadClass:
    def __init__(self, *, fps=60):
        self.thread = threading.Thread(target=self.thread_loop)
        self.lock = threading.Lock()
        self.active = False

        self.fps = fps

    def start(self):
        self.active = True
        self.thread.start()

    def stop(self):
        self.active = False

    def join(self):
        self.thread.join()

    def thread_loop(self):
        while self.active:
            self.step()
            time.sleep(1 / self.fps)

    def step(self):
        raise Exception("not implemented")


class IpCameraStreamReceiver(ThreadClass):
    def __init__(self, url):
        super().__init__(fps=1000)
        self.url = url
        self.frame = None

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame.copy() if frame is not None else None

    def get_frame(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        return frame

    def thread_loop(self):
        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            bytes = b''
            for chunk in response.iter_content(chunk_size=1024):
                bytes += chunk
                a = bytes.find(b'\xff\xd8')
                b = bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes[a:b+2]
                    bytes = bytes[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                                       cv2.IMREAD_COLOR)

                    self.set_frame(img)

                    with self.lock:
                        if not self.active:
                            break
        else:
            print("Failed to get video feed")


class CameraStreamReceiver(ThreadClass):
    def __init__(self):
        super().__init__(fps=120)
        self.frame = None
        self.camera = cv2.VideoCapture(0)

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame.copy() if frame is not None else None

    def get_frame(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        return frame

    def step(self):
        success, frame = self.camera.read()
        if not success:
            print('not success')
            return
        self.frame = frame


class MouseController(ThreadClass):
    def __init__(self):
        super().__init__(fps=120)

        self.mouse_active = False
        self.pos = None
        self.lmb = False
        self.rmb = False
        self.lmb_pressed = False
        self.rmb_pressed = False
        self.screen_wh = pyautogui.size()

    def deactivate(self):
        with self.lock:
            self.mouse_active = False

    def update(self, x: int, y: int, lmb: bool, rmb: bool):
        with self.lock:
            self.mouse_active = True
            self.pos = x, y
            self.lmb = lmb
            self.rmb = rmb

    def step(self):
        with self.lock:
            if not self.mouse_active or self.pos is None:
                return

            x, y = pyautogui.position()
            tox, toy = self.pos

            dist = np.sqrt((x - tox) ** 2 + (y - toy) ** 2)
            sx = x + (0.0005 * dist + 0.003) * (tox - x)
            sy = y + (0.0005 * dist + 0.003) * (toy - y)

            sx = min(max(sx, 1), self.screen_wh[0] - 1)
            sy = min(max(sy, 1), self.screen_wh[1] - 1)

            pyautogui.moveTo(sx, sy, _pause=False)

            if self.lmb and not self.lmb_pressed:
                pyautogui.mouseDown(button='left')
            if not self.lmb and self.lmb_pressed:
                pyautogui.mouseUp(button='left')
            self.lmb_pressed = self.lmb

            if self.rmb and not self.rmb_pressed:
                pyautogui.mouseDown(button='right')
            if not self.rmb and self.rmb_pressed:
                pyautogui.mouseUp(button='right')
            self.rmb_pressed = self.rmb


class ImageViewer(ThreadClass):
    def __init__(self):
        super().__init__(fps=30)
        self.image = None

    def start(self):
        super().start()

    def stop(self):
        super().stop()

    def join(self):
        super().join()

    def set_image(self, image):
        with self.lock:
            self.image = image.copy()

    def get_image(self):
        with self.lock:
            image = self.image.copy() if self.image is not None else None
        return image

    def thread_loop(self):
        while self.active:
            image = self.get_image()

            if image is not None:
                window_name = 'Video Stream'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 800, 600)
                cv2.imshow(window_name, image)

            if cv2.waitKey(1000 // self.fps) & 0xFF == ord('q'):
                self.stop()


class ImageProcessor(ThreadClass):
    def __init__(self):
        super().__init__()
        self.image = None
        self.receiver = CameraStreamReceiver()

        self.mouse_controller = MouseController()
        self.viewer = ImageViewer()

        self.hands = mp.solutions.hands.Hands(static_image_mode=False,
                                              max_num_hands=2)

    def start(self):
        self.receiver.start()
        self.mouse_controller.start()
        self.viewer.start()
        super().start()

    def stop(self):
        super().stop()
        self.viewer.stop()
        self.mouse_controller.stop()
        self.receiver.stop()

    def join(self):
        super().join()
        self.viewer.join()
        self.mouse_controller.join()
        self.receiver.join()

    def step(self):
        img = self.receiver.get_frame()
        if img is None:
            return

        img = cv2.flip(img, 1)

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        all_landmarks = result.multi_hand_landmarks

        if all_landmarks:
            for hand_landmarks in all_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        if all_landmarks and len(all_landmarks) == 1:
            hand_landmarks = all_landmarks[0]

            color = [255, 255, 255]
            if thumb_index := is_thumb_index_touching(hand_landmarks):
                color[2] = 0
            if thumb_mid := is_thumb_middle_touching(hand_landmarks):
                color[1] = 0

            img = draw_hand_middle(img, hand_landmarks, color)

            hand_x, hand_y = compute_hand_middle(img, hand_landmarks)

            rect_lt = (img.shape[1] // 4, img.shape[0] // 4)
            rect_rb = (img.shape[1] // 4 * 3, img.shape[0] // 4 * 3)
            cv2.rectangle(img, rect_lt, rect_rb, (0, 255, 0), 2)

            screen_x, screen_y = compute_screen_coords(hand_x, hand_y,
                                                       rect_lt, rect_rb)

            self.mouse_controller.update(screen_x, screen_y,
                                         thumb_index, thumb_mid)
        else:
            self.mouse_controller.deactivate()

        self.viewer.set_image(img)

        if not self.viewer.active:
            # case if quit-button pressed
            self.stop()
