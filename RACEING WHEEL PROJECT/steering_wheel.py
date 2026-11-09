import cv2
import mediapipe as mp
import numpy as np
import vgamepad as vg
import math
import time
import os

MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 0
STEERING_SENSITIVITY = 45
SMOOTHING = 0.15

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("hand_landmarker.task must be in the same folder.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

gamepad = vg.VX360Gamepad()
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Webcam not found.")

steering = 0.0
throttle = 0.0
neutral_angle = None
start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(image, timestamp)

        if result.hand_landmarks and len(result.hand_landmarks) >= 2:

            wrists = []

            for hand in result.hand_landmarks:
                wrist = hand[0]
                wrists.append((
                    int(wrist.x * width),
                    int(wrist.y * height)
                ))

            wrists.sort(key=lambda p: p[0])

            left_hand = wrists[0]
            right_hand = wrists[1]

            cv2.circle(frame, left_hand, 12, (0, 255, 0), -1)
            cv2.circle(frame, right_hand, 12, (0, 255, 0), -1)
            cv2.line(frame, left_hand, right_hand, (255, 0, 0), 3)

            dx = right_hand[0] - left_hand[0]
            dy = right_hand[1] - left_hand[1]

            angle = math.degrees(math.atan2(dy, dx))

            if neutral_angle is None:
                neutral_angle = angle

            difference = angle - neutral_angle

            target_steering = np.clip(
                difference / STEERING_SENSITIVITY,
                -1.0,
                1.0
            )

            steering += (target_steering - steering) * SMOOTHING

            target_throttle = np.clip(
                (height - right_hand[1]) / (height * 0.65),
                0.0,
                1.0
            )

            throttle += (target_throttle - throttle) * SMOOTHING

            gamepad.left_joystick_float(
                x_value=float(steering),
                y_value=0.0
            )

            gamepad.right_trigger(
                value=int(throttle * 255)
            )

            gamepad.update()

            cv2.putText(
                frame,
                f"Steering: {steering:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Throttle: {throttle:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "Show BOTH hands",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            steering *= 0.8
            throttle = 0.0

            gamepad.left_joystick_float(
                x_value=float(steering),
                y_value=0.0
            )

            gamepad.right_trigger(value=0)
            gamepad.update()

        cv2.imshow("Virtual Steering Wheel", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if result.hand_landmarks and len(result.hand_landmarks) >= 2:
                neutral_angle = angle
                print("Steering calibrated!")

        elif key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

gamepad.reset()
gamepad.update()