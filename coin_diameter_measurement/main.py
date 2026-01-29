import cv2
import numpy as np
import math

# ================= CONFIG =================
REFERENCE_COIN_DIAMETER_MM = 27.0   # ₹10 / ₹20
CAMERA_INDEX = 1
CIRCULARITY_THRESHOLD = 0.80

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
    blur = cv2.GaussianBlur(gray, (7, 7), 1.2)

    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles_data = []

    # -------- Detect circular objects ----------
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)

        if circularity >= CIRCULARITY_THRESHOLD:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circles_data.append((int(x), int(y), int(radius)))

    if len(circles_data) > 0:
        # Use largest circle as reference
        ref_circle = max(circles_data, key=lambda c: c[2])
        ref_radius_px = ref_circle[2]

        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * ref_radius_px)

        for (x, y, r) in circles_data:
            diameter_mm = 2 * r * mm_per_pixel

            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"Coin | {diameter_mm:.1f} mm",
                (x - 60, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )

            # Mark reference coin
            if r == ref_radius_px:
                cv2.putText(
                    frame,
                    "Reference Coin",
                    (x - 55, y + r + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

    cv2.imshow("Coin Diameter Measurement (Accurate)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
