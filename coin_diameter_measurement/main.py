import cv2
import numpy as np
import math

# ================= CONFIGURATION =================
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
CIRCULARITY_THRESHOLD = 0.78
MIN_CONTOUR_AREA = 2000

# ================= HELPER FUNCTIONS =================
def circularity(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0
    return (4 * math.pi * area) / (peri * peri)


def identify_coin(diameter_mm):
    for coin, ref in COIN_DIAMETERS_MM.items():
        if abs(diameter_mm - ref) <= TOLERANCE_MM:
            return coin
    return None


def classify_polygon(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        return "Square" if 0.95 <= ratio <= 1.05 else "Rectangle"
    return "Polygon"

# ================= CAMERA =================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'Q' to exit")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --------- COIN SCALE (REFERENCE) ----------
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1.2, 120,
        param1=150, param2=50,
        minRadius=30, maxRadius=120
    )

    mm_per_pixel = None
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        ref_circle = max(circles, key=lambda c: c[2])
        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * ref_circle[2])

    # --------- OBJECT ANALYSIS ----------
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        circ = circularity(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        label = ""
        color = (0, 255, 255)

        # --------- CIRCULAR OBJECT / COIN ----------
        if circ >= CIRCULARITY_THRESHOLD and mm_per_pixel is not None:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            diameter_mm = 2 * radius * mm_per_pixel
            coin = identify_coin(diameter_mm)

            if coin:
                label = f"{coin} Coin | {diameter_mm:.1f} mm"
                color = (0, 255, 0)
            else:
                label = f"Circular Object | {diameter_mm:.1f} mm"
                color = (255, 0, 0)

            cv2.circle(frame, (int(cx), int(cy)), int(radius), color, 2)

        # --------- NON-CIRCULAR SHAPES ----------
        else:
            shape = classify_polygon(cnt)
            label = shape
            cv2.drawContours(frame, [cnt], -1, color, 2)

        cv2.putText(
            frame, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, color, 2
        )

    cv2.imshow("Coin & Shape Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
