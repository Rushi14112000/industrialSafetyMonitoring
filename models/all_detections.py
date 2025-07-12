
import cv2
import torch
import math
import os
import requests
from datetime import datetime
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTForImageClassification
from ultralytics import YOLO
from torch.nn.functional import softmax
from collections import Counter

# === CONFIG ===
video_source = "C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoring/videos/all3detection.mp4"
image_save_path = "D:/Coding/IBM/images"
fire_alert_path = "D:/Coding/IBM/alert_fire"
handgest_alert_path = "D:/Coding/IBM/alert_danger"
backend_fire_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/firealert"
backend_gear_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/safetygear"
backend_hand_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/handgesture"
model_hand_gesture = "dima806/hand_gestures_image_detection"

os.makedirs(image_save_path, exist_ok=True)
os.makedirs(fire_alert_path, exist_ok=True)

# === Load Fire Detection Model ===
print("🔥 Loading fire model...")
fire_processor = AutoImageProcessor.from_pretrained("EdBianchi/vit-fire-detection")
fire_model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")
fire_model.eval()
fire_labels = ["Fire", "Nothing", "Smoke"]

# === Load Safety Gear Detection Model ===
print("🦺 Loading YOLO safety gear model...")
safety_model = YOLO("C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoring/models/yolo_models/best.pt")

# === Load Hand Gesture Detection Model ===
print("✋ Loading hand gesture model...")
hand_processor = AutoImageProcessor.from_pretrained(model_hand_gesture)
hand_model = ViTForImageClassification.from_pretrained(model_hand_gesture)
hand_model.eval()
hand_labels = hand_model.config.id2label
gesture_mapping = {
    "call": "call me 🤙",
    "rock": "fire or explosion risk 🤘",
    "spock": "chemical hazard alert 🖖"
}

# === Video Processing ===
cap = cv2.VideoCapture(video_source)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(total_frames / fps)
print(f"🎥 Video Duration: {duration} seconds, FPS: {fps}, Total Frames: {total_frames}")

for sec in range(duration):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to read frame at {sec}s")
        continue

    frame_path = os.path.join(image_save_path, f"frame_{sec}.jpg")
    cv2.imwrite(frame_path, frame)

    # === Fire Detection ===
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fire_inputs = fire_processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        fire_outputs = fire_model(**fire_inputs)
    fire_probs = softmax(fire_outputs.logits, dim=1).squeeze().tolist()
    fire_label = fire_labels[fire_probs.index(max(fire_probs))]

    print(f"[{sec}s] 🔥 Fire: {fire_label} | Probs: {fire_probs}")

    if fire_label == "Fire":
        fire_frame_path = os.path.join(fire_alert_path, f"fire_{sec}.jpg")
        cv2.imwrite(fire_frame_path, frame)
        fire_alert = {
        "camera_id": "camera_001",
        "video_link": fire_frame_path.replace("\\", "/"),
        "timestamp": datetime.now().isoformat()
        }
        try:
            res = requests.post(backend_fire_url, json=fire_alert)
            print("📡 Fire alert sent:", res.json() if res.headers.get('Content-Type') == 'application/json' else res.text)
        except Exception as e:
            print("❌ Fire alert error:", e)

    # === Safety Gear Detection ===
    safety_results = safety_model.predict(frame, verbose=False)
    detections = []
    for result in safety_results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = safety_model.names[cls_id]
            detections.append(class_name)

    gear_data = dict(Counter(detections))
    summary = ", ".join([f"{v} {k}" for k, v in gear_data.items()])
    print(f"{sec}: 640x640 {summary}")

    gear_body = {
        "camera-id": "camera_001",
        "data": gear_data,
        "timestamp": {
            "year": datetime.now().year,
            "month": datetime.now().month,
            "day": datetime.now().day,
            "hour": datetime.now().hour,
            "minute": datetime.now().minute,
            "second": datetime.now().second,
        },
    }

    try:
        r = requests.post(backend_gear_url, json=gear_body)
        print("✅ Safety gear alert sent:", r.text)
    except Exception as e:
        print("❌ Safety gear alert error:", e)

        # === Hand Gesture Detection ===
    hand_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_inputs = hand_processor(images=hand_rgb, return_tensors="pt")
    with torch.no_grad():
        hand_logits = hand_model(**hand_inputs).logits
    hand_probs = softmax(hand_logits, dim=1)[0]
    gesture_idx = torch.argmax(hand_probs).item()
    gesture_key = hand_labels[gesture_idx].lower().strip()
    confidence = float(hand_probs[gesture_idx])

    if confidence >= 0.5 and gesture_key in gesture_mapping:
        gesture_full = gesture_mapping[gesture_key]
        print(f"[{sec}s] ✋ Detected: {gesture_full} ({confidence:.2f})")

        gesture_img_path = os.path.join(handgest_alert_path, f"gesture_{gesture_key}_{sec}.jpg")
        cv2.imwrite(gesture_img_path, frame)

        gesture_payload = {
            "camera_id": "camera_001",
            "video_link": gesture_img_path.replace("\\", "/"),
            "gesture": gesture_full,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

        try:
            gr = requests.post(backend_hand_url, json=gesture_payload)
            print("📡 Hand gesture alert sent:", gr.json() if gr.headers.get('Content-Type') == 'application/json' else gr.text)
        except Exception as e:
            print("❌ Hand gesture alert error:", e)
    else:
        print(f"[{sec}s] ⚫ No relevant gesture detected or low confidence ({confidence:.2f})")

cap.release()
cv2.destroyAllWindows()
print("🎉 Detection pipeline complete.")
