import os
import cv2
import json
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    model = YOLO('weights/best.pt')
    results = []

    for img_name in sorted(os.listdir(args.input)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(args.input, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping invalid image: {img_name}")
            continue

        detections = model(img, device='cpu', conf=0.25, verbose=False)
        qrs = []
        for box in detections[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            qrs.append({"bbox": [x1, y1, x2, y2]})

        results.append({"image_id": image_id, "qrs": qrs})

    # ✅ Write all results at once (not inside the loop!)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved {len(results)} results to {args.output}")

if __name__ == "__main__":
    main()