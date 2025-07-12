# from PIL import Image
# import torch

# from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# import cv2
# from datetime import datetime, timedelta
# import time
# import requests

# import math
# import torchvision.transforms as transforms
# import ibm_boto3
# from ibm_botocore.client import Config, ClientError

# # NVVpHf46mQjkZObILVTUGuivEZSZI-JXyAbwhYG4FxtX
# credentials = {
#     "apikey": "P45M-20jSEkgxscQ5UpjIxbPgTdeHiocXeKG0vxY53af",
#     "iam_endpoint": "https://iam.cloud.ibm.com/v1/",
#     "auth_endpoint": "https://iam.cloud.ibm.com/v1/auth/token",
#     "region": "jp-tok",
#     "bucket": "ibmhacktesting1-donotdelete-pr-stnyxpwdejeura",
#     "cos_hmac_keys": {
#         "access_key_id": "bb1ac4d5f5f94e2982405d998dad104a",
#         "secret_access_key": "e6e0c3da8237c3a86e228793a71483c8ec1bb8dcd533102d",
#     },
# }

# # Create a Boto3 client object
# cos = ibm_boto3.client(
#     "s3",
#     ibm_api_key_id=credentials["apikey"],
#     ibm_service_instance_id="crn:v1:bluemix:public:cloud-object-storage:global:a/b0e7ae71a09c4e21b9a969e3570a3928:5c7f5575-2ad2-4adc-a509-0b1103eb6fe9::",
#     endpoint_url="https://s3.{}.cloud-object-storage.appdomain.cloud".format(
#         credentials["region"]
#     ),
#     config=Config(signature_version="oauth"),
#     aws_access_key_id=credentials["cos_hmac_keys"]["access_key_id"],
#     aws_secret_access_key=credentials["cos_hmac_keys"]["secret_access_key"],
# )

# preprocess = transforms.Compose(
#     [
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),  # Resize the image to match model's input size
#         transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         ),  # Normalize the image
#     ]
# )

# alert_count = 0


# def predict_image(frame_num):
#     extractor = AutoFeatureExtractor.from_pretrained("EdBianchi/vit-fire-detection")
#     model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")

#     model.eval()
#     # input_image = Image.open("C:/Users/shiva/Downloads/download.jpeg")
#     input_tensor = preprocess(frame_num)
#     input_batch = input_tensor.unsqueeze(0)
#     with torch.no_grad():
#         output = model(input_batch)

#     output_list = output[0].tolist()[0]
#     labels = ["Fire", "Nothing", "Smoke"]
#     final_probabilities = []
#     for i in output_list:
#         ex = math.exp(i)
#         final = ex / (1 + ex)
#         final_probabilities.append(final)

#     label_index = final_probabilities.index(max(final_probabilities))
#     return labels[label_index]


# def predict_image_func(video_source):
#     frames = []
#     alert_count = 0
#     video_count = 0
#     curr_frame = 0
#     cap = cv2.VideoCapture(video_source)
#     while cap.isOpened():
#         ret, frame = cap.read()

#         label = predict_image(frame)
#         if label == "Fire":
#             alert_count += 1
#         else:
#             alert_count = 0
#         if alert_count == 20:
#             print("Fire detected")
#             for i in range(21, 1, -1):
#                 frames.append(cv2.imread("D:/Coding/IBM/images/" + str(curr_frame - i) + ".jpg"))
#             frame_height, frame_width, _ = frames[0].shape
#             output_filename = f"D:/Coding/IBM/alert_fire/alert_{video_count}.mp4"
#             fourcc = cv2.VideoWriter_fourcc(*"avc1")
#             fps = 5
#             out = cv2.VideoWriter(
#                 output_filename, fourcc, fps, (frame_width, frame_height)
#             )
#             for frame in frames:
#                 out.write(frame)
#             out.release()
#             with open(f"D:/Coding/IBM/alert_fire/alert_{video_count}.mp4", "rb") as f:
#                 cos.upload_fileobj(
#                     f,
#                     credentials["bucket"],
#                     f"fire_alert_{video_count}.mp4",
#                 )
#             final_data = {
#                 "camera_id": "K4A6C7jaAiCwcQgg1LMv",
#                 "video_link": f"https://ibmhacktesting1-donotdelete-pr-stnyxpwdejeura.s3.jp-tok.cloud-object-storage.appdomain.cloud/fire_alert_{video_count}.mp4",
#                 "timestamp": {
#                     "year": datetime.now().year,
#                     "month": datetime.now().month,
#                     "day": datetime.now().day,
#                     "hour": datetime.now().hour,
#                     "minute": datetime.now().minute,
#                     "second": datetime.now().second,
#                 },
#             }
#             r = requests.post(
#                 "https://1b92-2409-40f2-310-b55b-1942-8bc7-7c3c-8e83.ngrok-free.app/firealert",
#                 json=final_data,
#             )
#             print(r.text)
#             video_count += 1
#             frames = []
#             alert_count = 0
#             exit()
#         curr_frame += 1
#     cap.release()








# from PIL import Image
# import torch
# from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# import cv2
# from datetime import datetime
# import math
# import torchvision.transforms as transforms
# import os

# # Preprocessing for input frames
# preprocess = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # Create required folders if not present
# os.makedirs("D:/Coding/IBM/images", exist_ok=True)
# os.makedirs("D:/Coding/IBM/alert_fire", exist_ok=True)


# def predict_image(frame):
#     extractor = AutoFeatureExtractor.from_pretrained("EdBianchi/vit-fire-detection")
#     model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")
#     model.eval()

#     input_tensor = preprocess(frame)
#     input_batch = input_tensor.unsqueeze(0)

#     with torch.no_grad():
#         output = model(input_batch)

#     output_list = output[0].tolist()[0]
#     labels = ["Fire", "Nothing", "Smoke"]

#     # Use sigmoid to convert logits to probabilities
#     final_probabilities = [math.exp(i) / (1 + math.exp(i)) for i in output_list]
#     label_index = final_probabilities.index(max(final_probabilities))
#     return labels[label_index]


# def predict_image_func(video_source):
#     frames = []
#     alert_count = 0
#     video_count = 0
#     curr_frame = 0

#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Failed to open video source.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Save current frame to disk for back-reference
#         frame_path = f"D:/Coding/IBM/images/{curr_frame}.jpg"
#         cv2.imwrite(frame_path, frame)

#         label = predict_image(frame)

#         if label == "Fire":
#             alert_count += 1
#         else:
#             alert_count = 0

#         if alert_count == 20:
#             print("🔥 Fire detected! Saving alert video...")

#             # Retrieve past 20 frames before detection
#             for i in range(21, 1, -1):
#                 path = f"D:/Coding/IBM/images/{curr_frame - i}.jpg"
#                 image = cv2.imread(path)
#                 if image is not None:
#                     frames.append(image)

#             if frames:
#                 height, width, _ = frames[0].shape
#                 output_path = f"D:/Coding/IBM/alert_fire/alert_{video_count}.mp4"
#                 fourcc = cv2.VideoWriter_fourcc(*"avc1")
#                 out = cv2.VideoWriter(output_path, fourcc, 5, (width, height))

#                 for f in frames:
#                     out.write(f)
#                 out.release()

#                 print(f"✅ Video saved locally at: {output_path}")

#                 # Optional: send POST request to your backend (you can remove this block)
#                 final_data = {
#                     "camera_id": "camera_001",
#                     "video_link": output_path,
#                     "timestamp": {
#                         "year": datetime.now().year,
#                         "month": datetime.now().month,
#                         "day": datetime.now().day,
#                         "hour": datetime.now().hour,
#                         "minute": datetime.now().minute,
#                         "second": datetime.now().second,
#                     },
#                 }

#                 try:
#                     import requests
#                     r = requests.post("https://e876-2409-40f2-3f-bf19-5941-9e32-ee1c-a38e.ngrok-free.app/firealert", json=final_data)
#                     print("✅ Alert sent to backend:", r.text)
#                 except Exception as e:
#                     print("⚠️ Failed to send alert:", e)

#                 video_count += 1
#                 frames = []
#                 alert_count = 0
#                 exit()

#         curr_frame += 1

#     cap.release()
#     cv2.destroyAllWindows()


# predict_image_func("C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/videos/fireVideo.mp4")




from PIL import Image
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import cv2
from datetime import datetime
import math
import torchvision.transforms as transforms
import os
import requests

# Preprocessing for input frames
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create required folders if not present
os.makedirs("D:/Coding/IBM/images", exist_ok=True)
os.makedirs("D:/Coding/IBM/alert_fire", exist_ok=True)

# Load model only once globally
processor = ViTImageProcessor.from_pretrained("EdBianchi/vit-fire-detection")
model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")
model.eval()
labels = ["Fire", "Nothing", "Smoke"]

def predict_image(frame):
    # Convert OpenCV frame (BGR) to RGB before passing to the processor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process image using ViTImageProcessor
    inputs = processor(images=frame_rgb, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0].tolist()

    # Convert logits to probabilities using softmax
    probabilities = [math.exp(i) / sum(math.exp(j) for j in logits) for i in logits]
    predicted_label = labels[probabilities.index(max(probabilities))]

    print(f"📷 Prediction: {predicted_label} | 🔢 Probabilities: {probabilities}")
    
    return predicted_label

def predict_image_func(video_source):
    frames = []
    alert_count = 0
    video_count = 0
    curr_frame = 0

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Failed to open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = f"D:/Coding/IBM/images/{curr_frame}.jpg"
        cv2.imwrite(frame_path, frame)

        label = predict_image(frame)
        print(f"📷 Frame {curr_frame}: {label}")

        if label == "Fire":
            alert_count += 1
        else:
            alert_count = 0

        if alert_count > 20:
            print("🔥 Fire detected! Saving alert video...")

            for i in range(21, 1, -1):
                path = f"D:/Coding/IBM/images/{curr_frame - i}.jpg"
                image = cv2.imread(path)
                if image is not None:
                    frames.append(image)

            if frames:
                height, width, _ = frames[0].shape
                output_path = f"D:/Coding/IBM/alert_fire/alert_{video_count}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                out = cv2.VideoWriter(output_path, fourcc, 5, (width, height))
                for f in frames:
                    out.write(f)
                out.release()

                print(f"✅ Video saved locally at: {output_path}")

                final_data = {
                    "camera_id": "camera_001",
                    "video_link": output_path,
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
                    r = requests.post(
                        "https://11e4-2409-40f2-12f-e57a-5866-dc90-d560-23aa.ngrok-free.app/firealert",
                        json=final_data
                    )
                    print("📡 Alert sent to backend:", r.text)
                except Exception as e:
                    print("⚠️ Failed to send alert:", e)

                video_count += 1
                frames = []
                alert_count = 0
                exit()

        curr_frame += 1

    cap.release()
    cv2.destroyAllWindows()

# Run the function
# if __name__ == "__main__":
#     video_path = "C:/Users/LENOVO/Desktop/FinalProject/IndustrialSafetyMonitoringFromGitHub/videos/fireVideo.mp4"
#     predict_image_func(video_path)
