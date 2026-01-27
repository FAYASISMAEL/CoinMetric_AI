import cv2
import numpy as np

# CONFIGURATION
# Standard Indian coin diameters (mm)
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

# HELPER FUNCTION
def identify_coin(diameter_mm):
    """
    Match measured diameter to nearest known coin diameter.
    """
    closest_coin = "Unknown"
    min_diff = float("inf")

    for coin, ref_diameter in COIN_DIAMETERS_MM.items():
        diff = abs(diameter_mm - ref_diameter)
        if diff < min_diff and diff <= TOLERANCE_MM:
            min_diff = diff
            closest_coin = coin

    return closest_coin

# CAMERA INITIALIZATION
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

        # Use LARGEST detected coin as reference
        reference_coin = max(circles, key=lambda c: c[2])
        ref_radius_px = reference_coin[2]

        mm_per_pixel = REFERENCE_COIN_DIAMETER_MM / (2 * ref_radius_px)

        for (x, y, r) in circles:
            x, y, r = int(x), int(y), int(r)

            diameter_mm = 2 * r * mm_per_pixel
            coin_name = identify_coin(diameter_mm)

            # Draw coin
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # Safe text position
            text_x = max(x - 50, 10)
            text_y = max(y - r - 10, 20)

            # Display coin name and diameter
            cv2.putText(
                frame,
                f"{coin_name} | {diameter_mm:.1f} mm",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # Display pixel radius (debug / viva proof)
            cv2.putText(
                frame,
                f"{r}px",
                (x - 20, y + r + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

            # Mark reference coin
            if r == ref_radius_px:
                cv2.putText(
                    frame,
                    "Reference Coin",
                    (x - 45, y + r + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

    cv2.imshow("Indian Coin Diameter Measurement (OpenCV)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP

cap.release()
cv2.destroyAllWindows()
