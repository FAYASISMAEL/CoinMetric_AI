import cv2
import numpy as np

# Configuration
REFERENCE_COIN_DIAMETER_MM = 27.0
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print('Error : Cannot open camera')
    exit()

print("Press 'Q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp = 1.2,
        minDist = 60,
        param1 = 100,
        param2 = 30,
        minRadius = 20,
        maxRadius = 120
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        ref_radius_px = circles[0][2]
        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * ref_radius_px)

        for (x, y, r) in circles:
            diameter_mm = 2 * r * mm_per_pixel

            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"{diameter_mm:.1f} mm",
                (x - 40, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
                )


        cv2.imshow('Coin Diameter Measurement (OpenCV)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindow()