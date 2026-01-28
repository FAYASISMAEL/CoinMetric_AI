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
MIN_CONTOUR_AREA = 1500

# HELPER FUNCTIONS
def identify_coin(diameter_mm):
    closest_coin = "Unknown Coin"
    min_diff = float("inf")

    for coin, ref in COIN_DIAMETERS_MM.items():
        diff = abs(diameter_mm - ref)
        if diff < min_diff and diff <= TOLERANCE_MM:
            min_diff = diff
            closest_coin = coin
    return closest_coin


def circularity(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0
    return (4 * math.pi * area) / (peri * peri)


def classify_polygon(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif vertices > 4:
        return "Polygon"
    return "Unknown Shape"

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
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # COIN DETECTION
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

    mm_per_pixel = None
    detected_coin_centers = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        reference_coin = max(circles, key=lambda c: c[2])
        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * reference_coin[2])

        for (x, y, r) in circles:
            detected_coin_centers.append((x, y))

            diameter_mm = 2 * r * mm_per_pixel
            coin_name = identify_coin(diameter_mm)

            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"{coin_name} | {diameter_mm:.1f} mm",
                (x - 70, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

    # SHAPE DETECTION (NON-COINS)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        circ = circularity(cnt)
        if circ >= CIRCULARITY_THRESHOLD:
            continue  # skip coins

        x, y, w, h = cv2.boundingRect(cnt)

        # Skip contours close to detected coins
        is_near_coin = False
        for (cx, cy) in detected_coin_centers:
            if abs(cx - (x + w // 2)) < 40 and abs(cy - (y + h // 2)) < 40:
                is_near_coin = True
                break
        if is_near_coin:
            continue

        shape_name = classify_polygon(cnt)

        cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
        cv2.putText(
            frame,
            shape_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.imshow("Coin & Shape Recognition (OpenCV)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
