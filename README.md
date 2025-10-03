# Multi-QR Code Recognition for Medicine Packs

This repository contains a complete solution for detecting multiple QR codes on medicine packaging images — built for the **Multi-QR Code Recognition Hackathon**.

## 🚀 Performance Highlights
- **mAP@0.5: 0.994** (99.4% detection accuracy)
- Trained on **100 annotated medicine pack images** with **359 QR codes** (avg. 3.6 QRs per image)
- Handles real-world challenges: **tilt, blur, partial occlusion, and lighting variations**
- Runs efficiently on **CPU** (no GPU required)

## 📦 Repository Structure
multiqr-hackathon/
├── infer.py # Main inference script
├── weights/best.pt # Trained YOLOv8n model (~6 MB)
├── requirements.txt # Python dependencies
├── outputs/ # Output JSON files
└── test_images/ # Sample test images (for demo)

## ⚙️ Setup
1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/multiqr-hackathon.git
   cd multiqr-hackathon

   python3.12 -m venv venv
source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt

Run Inference
To generate detection results on any folder of images:
python infer.py --input test_images/ --output outputs/submission_detection_1.json

📄 Output Format (Stage 1)
The output JSON follows the required schema:
[
  {
    "image_id": "img201",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
