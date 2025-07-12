# import cv2
# import torch
# import math
# import requests
# import os
# from PIL import Image
# from datetime import datetime
# from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# from transformers import AutoImageProcessor
# from ultralytics import YOLO
# import torchvision.transforms as transforms
# import time

# # === CONFIG ===
# video_source = "C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/videos/all_detection.mp4"
# image_save_path = "D:/Coding/IBM/images"
# alert_video_path = "D:/Coding/IBM/alert_fire"
# backend_fire_url = "https://11e4-2409-40f2-12f-e57a-5866-dc90-d560-23aa.ngrok-free.app/firealert"
# backend_gear_url = "https://11e4-2409-40f2-12f-e57a-5866-dc90-d560-23aa.ngrok-free.app/safetygear"

# os.makedirs(image_save_path, exist_ok=True)
# os.makedirs(alert_video_path, exist_ok=True)

# # === Load Fire Detection Model ===
# print("🔥 Loading fire model...")
# # fire_processor = AutoFeatureExtractor.from_pretrained("EdBianchi/vit-fire-detection")
# fire_processor = AutoImageProcessor.from_pretrained("EdBianchi/vit-fire-detection")
# fire_model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")
# fire_model.eval()
# fire_labels = ["Fire", "Nothing", "Smoke"]

# # === Load Safety Gear Detection Model ===
# print("🦺 Loading YOLO safety gear model...")
# safety_model = YOLO("C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/models/yolo_models/best.pt")

# # === Start Processing ===
# cap = cv2.VideoCapture(video_source)
# frame_num = 0
# alert_count = 0
# video_count = 0
# frames = []

# print("🚀 Starting detection pipeline...")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Save raw frame
#     frame_path = os.path.join(image_save_path, f"{frame_num}.jpg")
#     cv2.imwrite(frame_path, frame)

#     # ================= Fire Detection =================
#     fire_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     fire_inputs = fire_processor(images=fire_rgb, return_tensors="pt")
#     with torch.no_grad():
#         fire_outputs = fire_model(**fire_inputs)
#     logits = fire_outputs.logits[0].tolist()
#     probs = [math.exp(i) / sum(math.exp(j) for j in logits) for i in logits]
#     predicted_label = fire_labels[probs.index(max(probs))]
#     print(f"📷 Frame {frame_num}: Fire: {predicted_label} | Probabilities: {probs}")

#     if predicted_label == "Fire":
#         alert_count += 1
#     else:
#         alert_count = 0

#     if alert_count > 10:
#         print("🚨 Fire detected! Saving video clip...")
#         for i in range(21, 1, -1):
#             target_frame_num = frame_num - i
#             if target_frame_num < 0:
#                 continue

#             path = os.path.join(image_save_path, f"{target_frame_num}.jpg")
#             if not os.path.exists(path):
#                 print(f"⚠️ Frame {target_frame_num} not found: {path}")
#                 continue

#             img = cv2.imread(path)
#             if img is not None:
#                 frames.append(img)

#         if frames:
#             h, w, _ = frames[0].shape
#             output_path = os.path.join(alert_video_path, f"alert_{video_count}.mp4")
#             out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
#             for f in frames:
#                 out.write(f)
#             out.release()
#             print(f"✅ Fire alert video saved: {output_path}")

#             # Send alert to backend
#             fire_alert = {
#     "camera_id": "camera_001",
#     "video_link": output_path.replace("\\", "/"),
#     "timestamp": datetime.now().isoformat()  # 👈 already string, this is fine!
# }


#             try:
#                 res = requests.post(backend_fire_url, json=fire_alert)
#                 if res.headers.get('Content-Type') == 'application/json':
#                     print("📡 Fire alert sent:", res.json())
#                 else:
#                     print("⚠️ Backend returned non-JSON response:", res.text)
#             except Exception as e:
#                 print("⚠️ Failed to send fire alert:", e)

#             frames = []
#             alert_count = 0
#             video_count += 1

#     # ================= Safety Gear Detection =================
#     safety_results = safety_model.predict(frame)
#     counts = {}
#     for result in safety_results:
#         boxes = result.boxes.cpu().numpy()
#         for box in boxes:
#             cls = int(box.cls[0])
#             counts[cls] = counts.get(cls, 0) + 1

#     final = {safety_model.names[k]: counts.get(k, 0) for k in safety_model.names}

#     body = {
#         "camera-id": "camera_001",
#         "data": final,
#         "timestamp": {
#             "year": datetime.now().year,
#             "month": datetime.now().month,
#             "day": datetime.now().day,
#             "hour": datetime.now().hour,
#             "minute": datetime.now().minute,
#             "second": datetime.now().second,
#         },
#     }
#     try:
#         r = requests.post(backend_gear_url, json=body)
#         print("✅ Safety gear alert sent:", r.text)
#     except Exception as e:
#         print("⚠️ Failed to send safety gear alert:", e)

#     frame_num += 1
#     time.sleep(1)

# cap.release()
# cv2.destroyAllWindows()
# print("🎉 All detections complete.")



# last properly working code
# import cv2
# import torch
# import math
# import os
# import requests
# from datetime import datetime
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from ultralytics import YOLO
# from torch.nn.functional import softmax
# from collections import Counter

# # === CONFIG ===
# video_source = "C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/videos/all_detection.mp4"
# image_save_path = "D:/Coding/IBM/images"
# alert_video_path = "D:/Coding/IBM/alert_fire"
# backend_fire_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/firealert"
# backend_gear_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/safetygear"

# os.makedirs(image_save_path, exist_ok=True)
# os.makedirs(alert_video_path, exist_ok=True)

# # === Load Fire Detection Model ===
# print("🔥 Loading fire model...")
# fire_processor = AutoImageProcessor.from_pretrained("EdBianchi/vit-fire-detection")
# fire_model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")
# fire_model.eval()
# fire_labels = ["Fire", "Nothing", "Smoke"]

# # === Load Safety Gear Detection Model ===
# print("🦺 Loading YOLO safety gear model...")
# safety_model = YOLO("C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/models/yolo_models/best.pt")

# # === Video Processing ===
# cap = cv2.VideoCapture(video_source)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = int(total_frames / fps)
# print(f"🎥 Video Duration: {duration} seconds, FPS: {fps}, Total Frames: {total_frames}")

# for sec in range(duration):
#     cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # Seek to the N-th second
#     ret, frame = cap.read()
#     if not ret:
#         print(f"❌ Failed to read frame at {sec}s")
#         continue

#     # Save frame
#     frame_path = os.path.join(image_save_path, f"frame_{sec}.jpg")
#     cv2.imwrite(frame_path, frame)

#     # === Fire Detection ===
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     inputs = fire_processor(images=rgb, return_tensors="pt")
#     with torch.no_grad():
#         outputs = fire_model(**inputs)
#     probs = softmax(outputs.logits, dim=1).squeeze().tolist()
#     label = fire_labels[probs.index(max(probs))]

#     print(f"[{sec}s] 🔥 Fire: {label} | Probs: {probs}")

#     if label == "Fire":
#         fire_alert = {
#             "camera_id": "camera_001",
#             "video_link": frame_path.replace("\\", "/"),  # Send image path as 'video_link'
#             "timestamp": datetime.now().isoformat()
#         }
#         try:
#             res = requests.post(backend_fire_url, json=fire_alert)
#             if res.headers.get('Content-Type') == 'application/json':
#                 print("📡 Fire alert sent:", res.json())
#             else:
#                 print("⚠️ Fire alert non-JSON response:", res.text)
#         except Exception as e:
#             print("❌ Fire alert error:", e)

#     # === Safety Gear Detection ===
#     safety_results = safety_model.predict(frame, verbose=False)
#     detections = []

#     for result in safety_results:
#         boxes = result.boxes.cpu().numpy()
#         for box in boxes:
#             cls_id = int(box.cls[0])
#             class_name = safety_model.names[cls_id]
#             detections.append(class_name)

#     gear_data = dict(Counter(detections))

#     # Print terminal log in YOLO style
#     summary = ", ".join([f"{v} {k}" for k, v in gear_data.items()])
#     print(f"{sec}: 640x640 {summary}")

#     gear_body = {
#         "camera-id": "camera_001",
#         "data": gear_data,
#         "timestamp": {
#             "year": datetime.now().year,
#             "month": datetime.now().month,
#             "day": datetime.now().day,
#             "hour": datetime.now().hour,
#             "minute": datetime.now().minute,
#             "second": datetime.now().second,
#         },
#     }

#     try:
#         r = requests.post(backend_gear_url, json=gear_body)
#         print("✅ Safety gear alert sent:", r.text)
#     except Exception as e:
#         print("❌ Safety gear alert error:", e)

# cap.release()
# cv2.destroyAllWindows()
# print("🎉 Detection pipeline complete.")










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
video_source = "C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/videos/all_detection.mp4"
image_save_path = "D:/Coding/IBM/images"
alert_video_path = "D:/Coding/IBM/alert_fire"
backend_fire_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/firealert"
backend_gear_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/safetygear"
backend_hand_url = "https://3df1-2409-40f2-12f-e57a-38d0-e3e5-2576-28b6.ngrok-free.app/handgesture"
model_hand_gesture = "dima806/hand_gestures_image_detection"

os.makedirs(image_save_path, exist_ok=True)
os.makedirs(alert_video_path, exist_ok=True)

# === Load Fire Detection Model ===
print("🔥 Loading fire model...")
fire_processor = AutoImageProcessor.from_pretrained("EdBianchi/vit-fire-detection")
fire_model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")
fire_model.eval()
fire_labels = ["Fire", "Nothing", "Smoke"]

# === Load Safety Gear Detection Model ===
print("🦺 Loading YOLO safety gear model...")
safety_model = YOLO("C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/models/yolo_models/best.pt")

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
        fire_alert = {
            "camera_id": "camera_001",
            "video_link": frame_path.replace("\\", "/"),
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

        clip_path = os.path.join(alert_video_path, f"gesture_{gesture_key}_{sec}.mp4")
        h, w, _ = frame.shape
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (w, h))
        out.write(frame)
        out.release()

        gesture_payload = {
            "camera_id": "camera_001",
            "video_link": clip_path.replace("\\", "/"),
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
