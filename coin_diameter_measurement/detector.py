import cv2
import numpy as np

REFERENCE_COIN_DIAMETER_MM = 27.0
MIN_CONTOUR_AREA = 2500
CIRCULARITY_THRESHOLD = 0.75


def is_circle(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity >= CIRCULARITY_THRESHOLD


def detect_shape(cnt):
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        return "Square" if 0.95 <= ratio <= 1.05 else "Rectangle"

    else:
        return "Circle" if is_circle(cnt) else "Unknown"


def detect_objects(image_path):

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

    table = []
    circles = []

    # First pass: collect circles for scale reference
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        if is_circle(cnt):
            (_, _), r = cv2.minEnclosingCircle(cnt)
            circles.append(r)

    pixels_per_mm = None
    if circles:
        largest_radius = max(circles)
        pixels_per_mm = (2 * largest_radius) / REFERENCE_COIN_DIAMETER_MM

    # Second pass: detect & measure
    for cnt in contours:

        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        shape = detect_shape(cnt)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x, y, r = int(x), int(y), int(r)

        diameter_mm = "—"
        if shape == "Circle" and pixels_per_mm:
            diameter_mm = round((2 * r) / pixels_per_mm, 2)

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

        label = f"{shape}"
        if shape == "Circle":
            label += f" | {diameter_mm} mm"

        cv2.putText(
            output,
            label,
            (x - 60, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        table.append({
            "id": len(table) + 1,
            "shape": shape,
            "diameter": diameter_mm
        })

    cv2.putText(
        output,
        f"Objects Detected: {len(table)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    return output, table
