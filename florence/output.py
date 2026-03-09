import os
import json
from typing import Dict, List

import cv2
import numpy as np

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/result")


def quad_to_bbox_xyxy(quad: List[float]) -> List[int]:
    """Convert Florence quad_box [x1,y1,x2,y2,x3,y3,x4,y4] to axis-aligned bbox_xyxy."""
    xs = quad[0::2]
    ys = quad[1::2]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def deduplicate_detections(detections: List[Dict], containment_threshold: float = 0.8) -> List[Dict]:
    """
    Remove duplicate detections caused by tile overlap.
    If the smaller box has >= containment_threshold of its area inside the larger box, drop the smaller one.
    """
    def bbox_area(bbox):
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    def intersection_area(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    suppressed = set()
    for i, det_i in enumerate(detections):
        if i in suppressed:
            continue
        bbox_i = det_i.get("bbox_xyxy", [])
        if len(bbox_i) != 4:
            continue
        area_i = bbox_area(bbox_i)
        for j, det_j in enumerate(detections):
            if j <= i or j in suppressed:
                continue
            bbox_j = det_j.get("bbox_xyxy", [])
            if len(bbox_j) != 4:
                continue
            area_j = bbox_area(bbox_j)
            inter = intersection_area(bbox_i, bbox_j)
            smaller_area = min(area_i, area_j)
            if smaller_area > 0 and inter / smaller_area >= containment_threshold:
                suppressed.add(i if area_i <= area_j else j)

    return [det for k, det in enumerate(detections) if k not in suppressed]


def save_result(image_path: str, parsed: Dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(image_path))

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, f"{base}-florence-ocr.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    print(f"Saved: {json_path}")

    # Draw bboxes on the raw input image
    img = cv2.imread(image_path)
    for det in parsed.get("detections", []):
        quad = det.get("quad", [])
        bbox = det.get("bbox_xyxy", [])
        if len(quad) == 8:
            pts = np.array([[int(quad[i]), int(quad[i+1])] for i in range(0, 8, 2)], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
            label_x, label_y = int(quad[0]), max(int(quad[1]) - 4, 0)
        elif len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            label_x, label_y = x1, max(y1 - 4, 0)
        else:
            continue
        cv2.putText(img, det["text"], (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    bbx_path = os.path.join(OUTPUT_DIR, f"{base}-bbx{ext}")
    cv2.imwrite(bbx_path, img)
    print(f"Saved: {bbx_path}")
