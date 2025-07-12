from flask import Flask, request, jsonify, Response
import json

# from flask_sockets import Sockets
from flask_socketio import SocketIO
from flask_caching import Cache
import datetime
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter
from flask_cors import CORS



import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import numpy as np


config = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "filesystem",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 3000,
    "CACHE_DIR": Path("/tmp"),
    "CORS_HEADERS": "Content-Type",
}

cred = credentials.Certificate("ibm-safety-net-firebase-adminsdk.json")
firebase_app = firebase_admin.initialize_app(cred)
db = firestore.client()


app = Flask(__name__)
app.config.from_mapping(config)
ws = SocketIO(app)
ws.init_app(app, cors_allowed_origins="*")
cache = Cache(app)
# CORS(app)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

class Get_Data:
    def __init__(self, user_id, user_type):
        self.user_id = user_id
        self.user_type = user_type

    # monthly counts chaiye
    def sensor_data(self):
        
        ref = db.collection(self.user_type).document(self.user_id)
        data = ref.get().to_dict()
        if self.user_type == "managers":
            data_list = []
            d = {}
            l = []
            count = 0
            site_id = data["site_id"]
            data_ref = db.collection("sensor").where(filter = FieldFilter("site_id", "==", site_id)).get()
            for docs in data_ref:
                data_dict = {}
                readings = []
                time_str = []
                data = docs.to_dict()
                for i in data['data']:
                    readings.append(i['reading'])
                    time = datetime.fromtimestamp(i['timestamp'].timestamp())
                    time_str.append(time)
                    data_dict.update({"readings": readings, "timestamp": time_str})
                data_list.append(data_dict)
                for all in range(len(data_list)):
                    for item in range(len(data_list[all]["readings"])):
                        if data_list[all]['readings'][item] == True:
                            l.append(data_list[all]['timestamp'][item].month)
            l = np.array(l)
            un = np.unique(l)
            for ele in un:
                count = np.count_nonzero(l == ele)
                d[str(ele)] = count
            # print(d)
            return d
        # do for admin
        elif self.user_type == "admin":
            data_list = []
            l = []
            d = {}
            count = 0
            site_id = data["sites_id"]
            # print(site_id)
            for sites in site_id:
                data_ref = db.collection("sensor").where(filter = FieldFilter("site_id", "==", sites)).get()
                for docs in data_ref:
                    data_dict = {}
                    readings = []
                    time_str = []
                    data = docs.to_dict()
                    for i in data['data']:
                        readings.append(i['reading'])
                        time = datetime.fromtimestamp(i['timestamp'].timestamp())
                        time_str.append(time)
                        data_dict.update({"readings": readings, "timestamp": time_str})
                    data_list.append(data_dict)
                    for all in range(len(data_list)):
                        for item in range(len(data_list[all]["readings"])):
                            if data_list[all]['readings'][item] == True:
                                l.append(data_list[all]['timestamp'][item].month)
            l = np.array(l)
            un = np.unique(l)
            for ele in un:
                count = np.count_nonzero(l == ele)
                d[str(ele)] = count
            # print(d)
            return d
    
    # isme chaiye 1. safety score 2. camera_id se map karo safety score 3. time by time safety score ka graph for each camera 4. aggrgate safety score for each site
    def safety_gear_data(self):
        ref = db.collection(self.user_type).document(self.user_id)
        data = ref.get().to_dict()
        if self.user_type == "managers":
            data_list = []
            site_id = data["site_id"]
            avg = 0
            sc = 0
            data_ref = db.collection("safety-gear").where(filter = FieldFilter("site_id", "==", site_id)).get()
            for docs in data_ref:
                refer = db.collection("cameras").document(docs.id)
                camera_loc = refer.get().to_dict()['location']
                safety_score = []
                data_dict = {}
                score = 0
                for items in docs.to_dict()['data']:
                    
                    # score = (
                    #     int(items.get('safety-vest', 0)) +
                    #     int(items.get('hard-hat', 0)) +
                    #     int(items.get('mask', 0))
                    #     ) / (3 * max(1, int(items.get('person', 0))))
                    
                    person_count = int(items.get('person', 0))
                    if person_count == 0:
                        continue  # Skip frames with no people

                    score = (
                    int(items.get('safety-vest', 0)) +
                    int(items.get('hard-hat', 0)) +
                    int(items.get('mask', 0))
                    ) / (3 * person_count)


                    avg = avg + score
                    safety_score.append(score)
                for sc2 in safety_score:
                    sc = sc + sc2
                length = len(safety_score) if len(safety_score) != 0 else 1
                sc1 = sc/length
                data_dict.update({"location": camera_loc,"safety_score": round(sc1*100,2)})
                data_list.append(data_dict)
            avg = 0
            for dic in data_list:
                avg = avg + dic['safety_score']
            avg = avg/len(data_list)
            return [data_list,avg]
        # do for admin
        elif self.user_type == "admin":
            data_list = []
            site_id = data["sites_id"]
            for sites in site_id:
                avg = 0
                sc = 0
                sc3 = 0
                data_ref = db.collection("safety-gear").where(filter = FieldFilter("site_id", "==", sites)).get()
                for docs in data_ref:
                    # refer = db.collection("cameras").document(docs.id)
                    safety_score = []
                    data_dict = {}
                    score = 0
                    for items in docs.to_dict()['data']:
                        score = (int(items['safety-vest'])+int(items['hard-hat'])+int(items['mask']))/(3*int(items['person']))
                        avg = avg + score
                        safety_score.append(score)
                    for sc2 in safety_score:
                        sc = sc + sc2
                    length = len(safety_score) if len(safety_score) != 0 else 1
                    sc1 = sc/length
                    sc3 += sc1
                sc3 = sc3/len(data_ref)
                loc = db.collection("sites").document(sites).get().to_dict()['site_location']
                data_dict.update({"location": loc,"safety_score": round(sc3*100,2)})
                data_list.append(data_dict)
            avg = 0
            for dic in data_list:
                avg = avg + dic['safety_score']
            avgx = avg/len(data_list)
            # avg_dict = {"avg": avgx}
            return [data_list, avgx]

    def fire_stats(self):
        ref = db.collection(self.user_type).document(self.user_id)
        data = ref.get().to_dict()
        if self.user_type == "managers":
            l = []
            site_id = data["site_id"]
            duration_dict = {}
            data_ref = db.collection("fire-detection").where(filter = FieldFilter("site_id", "==", site_id)).get()
            for docs in data_ref:
                for ele in docs.to_dict()['data']:
                    time = datetime.fromtimestamp(ele['timestamp'].timestamp())
                    l.append(time.month)
            l = np.array(l)
            un = np.unique(l)
            for ele in un:
                count = np.count_nonzero(l == ele)
                duration_dict[str(ele)] = count
                # time_duration.append(duration)
            return duration_dict
        elif self.user_type == "admin":
            l = []
            site_id = data["sites_id"]
            duration_dict = {}
            for sites in site_id:
                data_ref = db.collection("fire-detection").where(filter = FieldFilter("site_id", "==", sites)).get()
                for docs in data_ref:
                    for ele in docs.to_dict()['data']:
                        time = datetime.fromtimestamp(ele['timestamp'].timestamp())
                        l.append(time.month)
            l = np.array(l)
            un = np.unique(l)
            for ele in un:
                count = np.count_nonzero(l == ele)
                duration_dict[str(ele)] = count
            return duration_dict
    
    def hand_gesture_data(self):
        ref = db.collection(self.user_type).document(self.user_id)
        data = ref.get().to_dict()
        duration_dict = {}
        if self.user_type == "managers":
            site_id = data["site_id"]
            data_ref = db.collection("hand-gesture").where(filter = FieldFilter("site_id", "==", site_id)).get()
            for docs in data_ref:
                time = datetime.fromtimestamp(docs.to_dict()['data'][-1]['timestamp'].timestamp())
                now = datetime.now()
                duration = (now - time).days
                # time_duration.append(duration)
                duration_dict[docs.id] = duration
            return duration_dict
        elif self.user_type == "admin":
            site_id = data["sites_id"]
            for sites in site_id:
                data_ref = db.collection("hand-gesture").where(filter = FieldFilter("site_id", "==", sites)).get()
                for docs in data_ref:
                    time = datetime.fromtimestamp(docs.to_dict()['data'][-1]['timestamp'].timestamp())
                    now = datetime.now()
                    duration = (now - time).days
                    duration_dict[docs.id] = duration
            return duration_dict
        

    def sensor_fire(self):
        sensor_data = self.sensor_data()
        fire_data = self.fire_stats()
        data_dict = {}
        keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        for key in keys:
            sens_acc = 0
            fire_acc = 0
            if key in sensor_data.keys() and key in fire_data.keys():
                fire_acc = fire_acc + fire_data[key]
                sens_acc = sens_acc + sensor_data[key]
                data_dict[str(key)] = {"sensor_accidents": sens_acc, "fire_accidents": fire_acc}
            elif key in sensor_data.keys():
                sens_acc = sens_acc + sensor_data[key]
                data_dict[str(key)] = {"sensor_accidents": sens_acc,"fire_accidents": fire_acc}
            elif key in fire_data.keys():
                fire_acc = fire_acc + fire_data[key]
                data_dict[str(key)] = {"sensor_accidents": sens_acc,"fire_accidents": fire_acc}
        return data_dict
    


def send_email(
    sender_email,
    sender_password,
    receiver_email,
    subject,
    message,
):
    # SMTP configuration for the email service
    smtp_host = "smtp.gmail.com"
    smtp_port = 587

    # Create a multipart message object
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # Attach the message body
    msg.attach(MIMEText(message, "plain"))
    # Create an SMTP session and start TLS for security
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.starttls()

    # Login to the email account
    server.login(sender_email, sender_password)

    # Send the email
    server.send_message(msg)

    # Terminate the SMTP session
    server.quit()


@ws.on("connect")
def connection():
    print("connected")


@ws.on("disconnect")
def disconnection():
    print("disconnected")


def send_alert(data):
    ws.emit(data)


@app.route("/getdata", methods=["GET"])
def get_data():
    output = {}
    try:
        user_id = request.args.get("user_id")
        user_type = request.args.get("user_type")

        ref = db.collection(user_type).document(user_id)
        data = ref.get().to_dict()

        def get_latest_entry(collection, site_id, cam_ref):
            ref = db.collection(collection).where("site_id", "==", site_id)
            result = []
            for doc in ref.get():
                temp = doc.to_dict()
                if "data" in temp and temp["data"]:
                    latest = sorted(temp["data"], key=lambda x: x["timestamp"])[-1]
                    entry = {
                        "camera_id": doc.id,
                        "site_id": site_id,
                        "timestamp": format_ist_timestamp(latest["timestamp"]),
                        "location": cam_ref.get(doc.id, {}).get("location", "")
                    }
                    if collection == "safety-gear":
                        latest.pop("timestamp")
                        entry["data"] = latest
                    else:
                        entry["video_link"] = latest["video_link"]
                    result.append(entry)
            return result

        if user_type == "managers":
            sites_ref = db.collection("sites").where(filter=firestore.FieldFilter("manager_id", "==", user_id))
            site = {}
            for sites in sites_ref.get():
                site.update(sites.to_dict())
                site.update({"site_id": sites.id})

            camera_ref = db.collection("cameras").where(
                filter=firestore.FieldFilter("site_id", "==", site["site_id"])).get()
            cam = []
            cam_ref = {}
            for camera in camera_ref:
                temp = camera.to_dict()
                cam_ref[camera.id] = temp
                temp.update({"camera_id": camera.id})
                cam.append(temp)
            output["camera_data"] = cam

            sensor_ref = db.collection("sensor").where(
                filter=firestore.FieldFilter("site_id", "==", site["site_id"])).get()
            sensor_data = []
            for sensor in sensor_ref:
                temp = sensor.to_dict()
                temp.pop("data", None)
                temp["sensor_id"] = sensor.id
                sensor_data.append(temp)
            output["sensor_data"] = sensor_data

            output["safety_gear_data"] = get_latest_entry("safety-gear", site["site_id"], cam_ref)
            output["fire_detection_data"] = get_latest_entry("fire-detection", site["site_id"], cam_ref)
            output["hand_gesture_data"] = get_latest_entry("hand-gesture", site["site_id"], cam_ref)

        elif user_type == "admin":
            sites_ref = db.collection("sites").where(filter=firestore.FieldFilter("admin_id", "==", user_id))
            sites = []
            site_ids = {}
            for site in sites_ref.get():
                temp = site.to_dict()
                temp["site_id"] = site.id
                site_ids[site.id] = temp
                sites.append(temp)
            output["sites"] = sites

            managers_ref = db.collection("managers").where(filter=firestore.FieldFilter("admin_id", "==", user_id))
            managers = []
            for manager in managers_ref.get():
                temp = manager.to_dict()
                temp["manager_id"] = manager.id
                managers.append(temp)
            output["managers"] = managers

            cameras_ref = db.collection("cameras").where(
                filter=firestore.FieldFilter("site_id", "in", list(site_ids.keys())))
            cam = []
            cam_ref = {}
            for camera in cameras_ref.get():
                temp = camera.to_dict()
                temp["camera_id"] = camera.id
                temp["location"] = f"{site_ids[temp['site_id']]['site_location']}, {temp['location']}"
                cam_ref[camera.id] = temp
                cam.append(temp)
            output["camera_data"] = cam

            sensor_ref = db.collection("sensor").where(
                filter=firestore.FieldFilter("site_id", "in", list(site_ids.keys())))
            sensor_data = []
            for sensor in sensor_ref.get():
                temp = sensor.to_dict()
                temp.pop("data", None)
                temp["sensor_id"] = sensor.id
                temp["location"] = f"{site_ids[temp['site_id']]['site_location']}, {temp['location']}"
                sensor_data.append(temp)
            output["sensor_data"] = sensor_data

            def get_admin_latest(collection):
                ref = db.collection(collection).where(
                    filter=firestore.FieldFilter("site_id", "in", list(site_ids.keys())))
                result = []
                for doc in ref.get():
                    temp = doc.to_dict()
                    if "data" in temp and temp["data"]:
                        latest = sorted(temp["data"], key=lambda x: x["timestamp"])[-1]
                        entry = {
                            "camera_id": doc.id,
                            "site_id": temp["site_id"],
                            "timestamp": format_ist_timestamp(latest["timestamp"]),
                            "location": cam_ref.get(doc.id, {}).get("location", "")
                        }
                        if collection == "safety-gear":
                            latest.pop("timestamp")
                            entry["data"] = latest
                        else:
                            entry["video_link"] = latest["video_link"]
                        result.append(entry)
                return result

            output["safety_gear_data"] = get_admin_latest("safety-gear")
            output["fire_detection_data"] = get_admin_latest("fire-detection")
            output["hand_gesture_data"] = get_admin_latest("hand-gesture")

        output["status"] = "success"
        return jsonify(output)

    except Exception as e:
        return jsonify({"status": str(e)}), 500


@app.route("/safetygear", methods=["POST"])
def safety_gear():
    try:
        data = request.json
        print(data)
        print("TYPE:", type(data))
        
        camera_id = data["camera-id"]
        timestamp = data["timestamp"]
        info = data["data"]

        print(info)

        timestamp_obj = get_timestamp_obj(timestamp)

        temp = {
            "hard-hat": info.get("Hardhat", 0),
            "no-hard-hat": info.get("NO-Hardhat", 0),
            "mask": info.get("Mask", 0),
            "no-mask": info.get("NO-Mask", 0),
            "safety-vest": info.get("Safety Vest", 0),
            "no-safety-vest": info.get("NO-Safety Vest", 0),
            "person": info.get("Person", 0),
            "timestamp": timestamp_obj,
        }

        doc_ref = db.collection("safety-gear").document(camera_id)

        # First try update, fallback to set if not exists
        try:
            doc_ref.update({"data": firestore.ArrayUnion([temp])})
        except Exception as e:
            # Create document with initial data array
            doc_ref.set({"data": [temp]}, merge=True)

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": f"error {str(e)}"}), 500



@app.route("/handgesture", methods=["POST"])
def hand_gesture():
    # timestamp-{},video-link-string, camera-id-string
    global global_data
    try:
        data = request.json
        camera_id = data["camera_id"]
        timestamp = data["timestamp"]
        video_link = data["video_link"]

        timestamp_obj = get_timestamp_obj(timestamp)

        doc_ref = db.collection("hand-gesture").document(camera_id)
        doc_ref.update(
            {
                "data": firestore.ArrayUnion(
                    [{"timestamp": timestamp_obj, "video_link": video_link}]
                )
            }
        )

        camera_ref = db.collection("cameras").document(camera_id)
        camera_data = camera_ref.get().to_dict()

        data.update({"location": camera_data["location"]})

        data.update({"type": "hand-gesture"})
        data.update({"timestamp": timestamp_obj.strftime("%d/%m/%Y %H:%M:%S")})

        ws.emit("notification", data)
        send_email(sender_email='rushirpatil14@gmail.com',sender_password='ywuc gtpa khwy loty',receiver_email='rushirpatil491@gmail.com',subject='Person in Danger!',message=f'A person is in danger at location {camera_data["location"]}. Please check the video at {video_link}')

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": f"error {str(e)}"}), 500


@app.route("/firealert", methods=["POST"])
def fire_alert():
    # timestamp-{},video-link-string, camera-id-string
    try:
        data = request.json
        print("Received Data:", data)

        camera_id = data["camera_id"]
        timestamp = data["timestamp"]
        video_link = data["video_link"]
        

        timestamp_obj = get_timestamp_obj(timestamp)

        doc_ref = db.collection("fire-detection").document(camera_id)
        doc_ref.update(
            {
                "data": firestore.ArrayUnion(
                    [{"timestamp": timestamp_obj, "video_link": video_link}]
                )
            }
        )

        camera_ref = db.collection("cameras").document(camera_id)
        camera_data = camera_ref.get().to_dict()

        data.update({"location": camera_data["location"]})

        data.update({"type": "fire"})
        data.update({"timestamp": timestamp_obj.strftime("%d/%m/%Y %H:%M:%S")})

        ws.emit("notification", data)
        send_email(sender_email='rushirpatil14@gmail.com',sender_password='ywuc gtpa khwy loty',receiver_email='rushirpatil491@gmail.com',subject='Fire Alert!',message=f'There is a fire at location {camera_data["location"]}. Please check the video at {video_link}')

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": f"error {str(e)}"}), 500


# @app.route("/sensor", methods=["POST"])
# def read_sensor():
#     # sensor-id-string, reading-bool, timestamp-{}
#     try:
#         data = request.json
#         doc_id = data["sensor-id"]
#         reading = data["reading"]
#         timestamp = data["timestamp"]
#         timestamp_obj = get_timestamp_obj(timestamp)
#         doc_ref = db.collection("sensor").document(doc_id)
#         doc_ref.update(
#             {
#                 "data": firestore.ArrayUnion(
#                     [{"timestamp": timestamp_obj, "reading": reading}]
#                 )
#             }
#         )
#         if reading == True:
#             if cache.get("sensor-cache") == None:
#                 cache.set("sensor-cache", 1)
#             else:
#                 if cache.get("sensor-cache") >= 5:
#                     sensor_ref = db.collection("sensor").document(doc_id)
#                     sensor_data = sensor_ref.get().to_dict()

#                     data.update({"location": sensor_data["location"]})
#                     data["type"] = "sensor"
#                     data.update(
#                         {"timestamp": timestamp_obj.strftime("%d/%m/%Y %H:%M:%S")}
#                     )

#                     ws.emit("notification", data)
#                     send_email(sender_email='rushirpatil14@gmail.com',sender_password='ywuc gtpa khwy loty',receiver_email='rushirpatil491@gmail.com',subject='Gas Leak!',message=f'Gas leak at location {sensor_data["location"]}')
#                     cache.set("sensor-cache", 0)
#                 cache.set("sensor-cache", cache.get("sensor-cache") + 1)
#         else:
#             cache.set("sensor-cache", 0)

#         return jsonify({"status": "success"})
#     except Exception as e:
#         return jsonify({"status": f"error {str(e)}"}), 500


from datetime import datetime, timedelta, timezone

def get_timestamp_obj(timestamp):
    ist = timezone(timedelta(hours=5, minutes=30), "IST")
    
    if isinstance(timestamp, dict):
        return datetime(
            timestamp["year"],
            timestamp["month"],
            timestamp["day"],
            timestamp["hour"],
            timestamp["minute"],
            timestamp["second"],
            tzinfo=ist,
        )
    elif isinstance(timestamp, str):
        return datetime.fromisoformat(timestamp).replace(tzinfo=ist)
    else:
        raise ValueError("Invalid timestamp format")
    
def format_ist_timestamp(timestamp):
    ist = timezone(timedelta(hours=5, minutes=30), "IST")

    if isinstance(timestamp, datetime):
        if timestamp.tzinfo:
            dt = timestamp.astimezone(ist)
        else:
            dt = timestamp.replace(tzinfo=timezone.utc).astimezone(ist)
        return dt.strftime("%#d %B %Y at %H:%M:%S UTC%z")
    else:
        raise ValueError("Invalid timestamp format")



@app.route("/getstats", methods=["GET"])
def statistics():
    user_id = request.args.get("user_id")
    user_type = request.args.get("user_type")
    data = Get_Data(user_id, user_type)
    sensor_data = data.sensor_data()
    safety_gear_data,avg_data = data.safety_gear_data()
    fire_data = data.fire_stats()
    hand_gesture_data = data.hand_gesture_data()
    sensor_fire_data = data.sensor_fire()

    return jsonify({ "safety_gear_data": safety_gear_data, "sensor_fire_data": sensor_fire_data,'safety_avg_data':avg_data})


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == '__main__':
    ws.run(app, host='0.0.0.0', port=5000)

