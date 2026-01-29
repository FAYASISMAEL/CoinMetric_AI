import cv2
import numpy as np
import math

# ================= CONFIGURATION =================
REFERENCE_COIN_DIAMETER_MM = 27.0     # ₹10 / ₹20 coin
CAMERA_INDEX = 1
CIRCULARITY_THRESHOLD = 0.75
MIN_CONTOUR_AREA = 1500

# ================= HELPER FUNCTIONS =================
def is_circle(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False, 0

    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    return circularity >= CIRCULARITY_THRESHOLD, circularity


# ================= CAMERA =================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Error: Cannot open camera")
    exit()

print("Press 'Q' to exit")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)

    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # -------- Detect reference coin using Hough --------
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

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        ref_circle = max(circles, key=lambda c: c[2])
        ref_radius_px = ref_circle[2]
        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * ref_radius_px)

        cv2.circle(frame, (ref_circle[0], ref_circle[1]), ref_radius_px,
                   (0, 255, 0), 3)
        cv2.putText(
            frame,
            "Reference Coin",
            (ref_circle[0] - 50, ref_circle[1] + ref_radius_px + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # -------- Shape Detection --------
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        is_circ, circ_value = is_circle(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if mm_per_pixel is not None:
            width_mm = w * mm_per_pixel
            height_mm = h * mm_per_pixel
        else:
            width_mm = height_mm = 0

        # ---------- Circle / Coin ----------
        if is_circ:
            diameter_mm = max(width_mm, height_mm)

            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Circle | Dia: {diameter_mm:.1f} mm",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 100, 250),
                2
            )

        # ---------- Rectangle / Square / Triangle ----------
        else:
            aspect_ratio = w / float(h)

            if 0.9 <= aspect_ratio <= 1.1:
                shape_name = "Square"
            elif aspect_ratio > 1.2 or aspect_ratio < 0.8:
                shape_name = "Rectangle"
            else:
                shape_name = "Triangle / Irregular"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{shape_name} {width_mm:.1f}x{height_mm:.1f} mm",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    cv2.imshow("Coin & Shape Measurement System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
