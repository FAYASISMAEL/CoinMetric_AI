import cv2
import numpy as np
import math

# CONFIGURATION
COIN_DIAMETERS_MM = {
    "₹1": 20.0,
    "₹2": 23.0,
    "₹5": 25.0,
    "₹10": 27.0,
    "₹20": 27.0
}

REFERENCE_COIN_DIAMETER_MM = 27.0
CAMERA_INDEX = 1
TOLERANCE_MM = 1.5
CIRCULARITY_THRESHOLD = 0.75

# HELPER FUNCTIONS
def identify_coin(diameter_mm):
    closest_coin = "Unknown"
    min_diff = float("inf")

    for coin, ref_diameter in COIN_DIAMETERS_MM.items():
        diff = abs(diameter_mm - ref_diameter)
        if diff < min_diff and diff <= TOLERANCE_MM:
            min_diff = diff
            closest_coin = coin

    return closest_coin


def is_circle(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False, 0

    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    return circularity >= CIRCULARITY_THRESHOLD, circularity

# CAMERA
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'Q' to exit")

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

    # Edge detection for contour-based shape recognition
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=150,
        param2=50,
        minRadius=30,
        maxRadius=120
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        reference_coin = max(circles, key=lambda c: c[2])
        ref_radius_px = reference_coin[2]
        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * ref_radius_px)

        for (x, y, r) in circles:
            x, y, r = int(x), int(y), int(r)

            # Match contour near detected circle
            shape_label = "Unknown Shape"
            circ_value = 0

            for cnt in contours:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                if abs(cx - x) < 10 and abs(cy - y) < 10:
                    is_circ, circ_value = is_circle(cnt)
                    shape_label = "Circle (Coin)" if is_circ else "Non-Circular"
                    break

            diameter_mm = 2 * r * mm_per_pixel
            coin_name = identify_coin(diameter_mm)

            # Draw circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # Display text
            text_x = max(x - 70, 10)
            text_y = max(y - r - 15, 25)

            cv2.putText(
                frame,
                f"{coin_name} | {diameter_mm:.1f} mm",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"{shape_label}",
                (text_x, text_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (3, 24, 255),
                1
            )

            cv2.putText(
                frame,
                f"Circ: {circ_value:.2f}",
                (text_x, text_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 47, 191),
                1
            )

    cv2.imshow("Coin Diameter & Shape Recognition (OpenCV)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
