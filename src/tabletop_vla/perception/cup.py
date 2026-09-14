"""Calibrated color baseline; explicitly not a VLM or arbitrary-object detector."""

import numpy as np


def locate_blue_cup(rgb, *, camera_height=2.45, fovy_degrees=48, cup_top_z=0.785):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB camera pixels")
    red, green, blue = np.moveaxis(image.astype(float), -1, 0)
    # Both blue and green can saturate under bright randomized lighting. The
    # calibrated pickup ROI excludes the similarly saturated bottle and markers.
    pixels_to_m = 2 * (camera_height - cup_top_z) * np.tan(np.deg2rad(fovy_degrees) / 2) / image.shape[0]
    xs = (np.arange(image.shape[1]) - (image.shape[1] - 1) / 2) * pixels_to_m
    ys = ((image.shape[0] - 1) / 2 - np.arange(image.shape[0])) * pixels_to_m
    region = ((xs > 0.12) & (xs < 0.32))[None, :] & ((ys > -0.14) & (ys < 0.02))[:, None]
    mask = (blue > 1.25 * red) & (blue >= green - 3) & (green > red + 25) & (blue > 55) & region
    rows, cols = np.where(mask)
    if len(rows) < 30:
        raise ValueError("No sufficiently visible blue cup in the camera image")
    bounds = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
    width, height = bounds[2] - bounds[0] + 1, bounds[3] - bounds[1] + 1
    density = len(rows) / (width * height)
    if max(width, height) > image.shape[0] * 0.15 or density < 0.15:
        raise ValueError("Ambiguous blue regions; cannot ground one cup safely")
    u, v = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    x = (u - (image.shape[1] - 1) / 2) * pixels_to_m
    y = ((image.shape[0] - 1) / 2 - v) * pixels_to_m
    if not 0.12 < x < 0.32 or not -0.14 < y < 0.02:
        raise ValueError("Detected cup is outside the validated pickup region")
    return {"object": "blue_cup", "xy": [float(x), float(y)],
            "pixel_bounds": bounds, "visible_pixels": len(rows),
            "estimated_width_y_m": float(height * pixels_to_m),
            "method": "color segmentation + calibrated top camera + nominal cup-top plane",
            "limitations": "One visible blue cup, known camera/table, bounded cup dimensions; not a VLM"}
