import cv2

def draw_axes(image, color=(0, 255, 0)):
    """
    Draw X and Y axes (centered) on the given image.
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Draw horizontal (X-axis) line
    cv2.line(image, (0, center_y), (w, center_y), color, 2)

    # Draw vertical (Y-axis) line
    cv2.line(image, (center_x, 0), (center_x, h), color, 2)
