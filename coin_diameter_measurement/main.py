import cv2
import numpy as np
import math

# CONFIG
REFERENCE_COIN_DIAMETER_MM = 27.0
CAMERA_INDEX = 1
CIRCULARITY_THRESHOLD = 0.78

# CAMERA
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'Q' to exit")

mm_per_pixel = None

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.2)
    edges = cv2.Canny(blurred, 60, 160)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # COIN DETECTION
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=120,
        param2=45,
        minRadius=30,
        maxRadius=130
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        # Largest coin as reference
        ref_coin = max(circles, key=lambda c: c[2])
        ref_radius_px = ref_coin[2]
        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * ref_radius_px)

        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(
                frame, "Coin",
                (x - 25, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2
            )

    # SHAPE & DIMENSIONS
    if mm_per_pixel:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)

            width_mm = w * mm_per_pixel
            height_mm = h * mm_per_pixel

            shape_name = None
            dimension_text = ""

            # CIRCLE (COIN CONFIRMATION)
            circularity = (4 * math.pi * area) / (peri * peri) if peri else 0
            if circularity >= CIRCULARITY_THRESHOLD:
                shape_name = "Circular Object"
                diameter_mm = (w + h) / 2 * mm_per_pixel
                dimension_text = f"D: {diameter_mm:.1f} mm"

            # TRIANGLE
            elif len(approx) == 3:
                shape_name = "Triangle"
                dimension_text = f"Base: {width_mm:.1f} mm | Height: {height_mm:.1f} mm"

            # RECTANGLE / SQUARE
            elif len(approx) == 4:
                aspect_ratio = w / float(h)

                if 0.95 <= aspect_ratio <= 1.05:
                    shape_name = "Square"
                    dimension_text = f"Side: {width_mm:.1f} mm"
                else:
                    shape_name = "Rectangle"
                    dimension_text = f"L: {width_mm:.1f} mm | W: {height_mm:.1f} mm"

            else:
                continue

            # DRAW
            cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)

            cv2.putText(
                frame,
                shape_name,
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                dimension_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

    cv2.imshow("Shape & Dimension Measurement (OpenCV)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
