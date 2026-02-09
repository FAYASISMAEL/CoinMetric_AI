import cv2
import numpy as np
import math

REFERENCE_COIN_DIAMETER_MM = 27.0
CIRCULARITY_THRESHOLD = 0.82
MIN_CONTOUR_AREA = 2500


def is_circle(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False, 0

    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    return circularity >= CIRCULARITY_THRESHOLD, circularity


def process_image(input_path, output_path):
    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError("Unable to read image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 1.5)

    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        is_circ, circ = is_circle(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        if is_circ:
            coin_count += 1
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)

            text = f"Coin {coin_count}"
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Summary banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Total Coins Detected: {coin_count}",
        (20, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (0, 255, 255),
        2
    )

    cv2.imwrite(output_path, frame)
