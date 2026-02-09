import cv2
import numpy as np

REFERENCE_COIN_DIAMETER_MM = 27.0
MIN_CONTOUR_AREA = 2500
CIRCULARITY_THRESHOLD = 0.70


def is_circle(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= CIRCULARITY_THRESHOLD


def detect_coins(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 3
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    coins = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        if not is_circle(cnt):
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        coins.append((int(x), int(y), int(radius)))

    if not coins:
        return output, []

    reference = max(coins, key=lambda c: c[2])
    pixels_per_mm = (2 * reference[2]) / REFERENCE_COIN_DIAMETER_MM

    table = []

    for idx, (x, y, r) in enumerate(coins, start=1):
        diameter_mm = round((2 * r) / pixels_per_mm, 2)

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{diameter_mm} mm",
            (x - 40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        table.append({
            "id": idx,
            "diameter": diameter_mm
        })

    cv2.putText(
        output,
        f"Coins Detected: {len(table)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    return output, table
