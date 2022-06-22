import io
import base64
import numpy as np
from flask import Flask, render_template, Response, request
import cv2
import time
from PIL import Image
import torch
import torchvision.transforms as tt
import torch.nn.functional as F
from training.DeepEmotion import DeepEmotion
from matplotlib.figure import Figure
import warnings
import mediapipe as mp
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch.nn.functional")

facec = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

models = {
    "1": [DeepEmotion(), "DeepEmotion_FERG.pt", tt.Compose([tt.Resize((48, 48)), tt.Grayscale(num_output_channels=1), tt.ToTensor(), tt.Normalize((0.5,), (0.5,))]), (255, 214, 165)],
    "2": [DeepEmotion(), "DeepEmotion_FERG.pt", tt.Compose([tt.Resize((48, 48)), tt.Grayscale(num_output_channels=1), tt.ToTensor(), tt.Normalize((0.5,), (0.5,))]), (255, 214, 165)],
    "3": [DeepEmotion(), "DeepEmotion_FERG.pt", tt.Compose([tt.Resize((48, 48)), tt.Grayscale(num_output_channels=1), tt.ToTensor(), tt.Normalize((0.5,), (0.5,))]), (255, 214, 165)],
    "4": [DeepEmotion(), "DeepEmotion_FERG.pt", tt.Compose([tt.Resize((48, 48)), tt.Grayscale(num_output_channels=1), tt.ToTensor(), tt.Normalize((0.5,), (0.5,))]), (255, 214, 165)],

}

application = Flask(__name__, template_folder='./')
pause = "false"
stream_mode = "video"
current_model = "1"
current_video = "1"
emotion = {0: 'angry', 1: 'disgust', 2: 'fear',
           3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
start = time.time()
videoname = "video/1.mp4"
video = cv2.VideoCapture(videoname)
sleep = True
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
score = torch.tensor(
    [[-0.9674,  1.9319, -1.7264, -2.9173, -0.5697,  2.3320, -1.4329]]).cpu()  # default
transform = None
with open("static/images/error.jpeg", "rb") as image:
    f = image.read()
    last_frame = bytes(f)
with open("static/images/defaultframe.jpeg", "rb") as image:
    f = image.read()
    current_frame = bytes(f)
frame_queue = []  # synchronizing graph plot and frame plot (delay)
frame_queue.append(current_frame)


def set_model(x):
    global sleep, transform, model_setup, model_name, model
    sleep = True
    model_setup = models[x]
    model = model_setup[0]
    model.cpu()
    model.eval()
    model_name = model_setup[1].split("_")[0]
    model.load_state_dict(torch.load(
        "training/Checkpoints/{}".format(model_setup[1]), map_location=torch.device('cpu')))
    transform = model_setup[2]


set_model(current_model)


def loadvideo(x):
    global videoname
    global video
    global start
    start = time.time()
    videoname = x
    video = cv2.VideoCapture(videoname)


def predict(img):
    global score, sleep
    if not sleep:
        img = Image.fromarray(img)
        img_normalized = transform(img).float()
        img_normalized = img_normalized.unsqueeze_(0)
        with torch.no_grad():
            output = model(img_normalized)
            _, index = torch.max(output, 1)
            score = output
            class_name = emotion[int(index[0])]
            return class_name


def redirect_plot():
    while True:
        if pause == "true":
            continue
        else:
            try:
                class_names = ['Angry', 'Disgust', 'Fear',
                               'Happy', 'Sad', 'Surprise', 'Neutral']
                fig = Figure()
                softmax = F.softmax(score, dim=1)
                data = softmax[0].data.cpu().numpy()
                width = 0.8
                color_list = ['red', 'orangered', 'darkorange',
                              'limegreen', 'darkgreen', 'royalblue', 'navy']
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(
                    "Probabilities of classification in % (with 0.1ms delay)", fontsize=16)
                ax.set_xlabel("Emotion", fontsize=16)
                ax.set_ylabel("Score", fontsize=16)
                ax.set_ylim(0, 1)
                for i in range(len(class_names)):
                    x = ax.bar(class_names[i], data[
                        i], width, color=color_list[i])
                    ax.text(x[0].get_x() + x[0].get_width() / 2.0, x[0].get_height(),
                            f'{x[0].get_height()*100:.2f}%', ha='center', va='bottom')
                fig.savefig("static/images/plot.jpeg")
            except Exception:
                print("Error! Skip this plot and show last one..")
            with open("static/images/plot.jpeg", "rb") as image:
                f = image.read()
                frame = bytes(f)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def redirect_frame():
    global sleep
    global start
    global current_frame
    global last_frame
    global frame_queue
    while True:
        if pause == "true":
            continue
        else:
            try:
                if sleep:
                    sleep = False
                    time.sleep(3)
                if time.time()-start >= 28 and videoname != 0:
                    loadvideo(videoname)
                if videoname == 0:
                    fr = current_frame
                else:
                    _, fr = video.read()
                faces = facec.detectMultiScale(fr, 1.3, 5)
                height, width, _ = fr.shape
                for (x, y, w, h) in faces:
                    fc = fr[y: y+h, x: x+w]
                    pred = predict(fc)
                    fontsize = round((height*width)*0.0000020)
                    cv2.putText(fr, pred, (x+(w//3), y-15),
                                font, fontsize+1, (255, 255, 255), 2)
                    cv2.putText(fr, f"X: {x} Y: {y}", (20, height-height//16),
                                font, fontsize, (255, 255, 255), 2)
                    xc = int((x + x+w)/2)
                    yc = int((y + y+h)/2)
                    radius = int(w/2)
                    cv2.circle(fr, (xc, yc), radius,
                               models[current_model][3], 2)
                    break
                rgb_image = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb_image)
                if result.multi_face_landmarks:
                    for facial_landmarks in result.multi_face_landmarks:
                        for i in range(0, 468):
                            pt1 = facial_landmarks.landmark[i]
                            x = int(pt1.x * width)
                            y = int(pt1.y * height)
                            cv2.circle(fr, (x, y),
                                       2, (255, 255, 255), -1)
                _, jpeg = cv2.imencode('.jpg', fr)
                frame = jpeg.tobytes()
                last_frame = frame  # exception handler frame
                frame_queue.append(frame)  # queue
                frame = frame_queue.pop(0)
            except Exception:
                frame = last_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@ application.route('/')
def index():
    return render_template('index.html')


@application.route('/get_frame', methods=['POST'])
def get_image():
    global current_frame
    image_b64 = request.values['imageBase64']
    imgdata = image_b64.split(',')[1]
    decoded = base64.b64decode(imgdata)
    current_frame = np.array(Image.open(io.BytesIO(decoded)).convert('RGB'))
    return ''


@ application.route('/video_feed')
def video_feed():
    return Response(redirect_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@ application.route('/graph_plot')
def graph_plot():
    return Response(redirect_plot(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@ application.route('/pause', methods=['POST'])
def get_pause():
    global pause
    pause = request.form['status']
    return pause


@ application.route('/stream_mode', methods=['POST', 'GET'])
def get_stream_mode():
    global stream_mode
    global current_video
    stream_mode = request.form['status']
    if stream_mode == 'webcam':
        loadvideo(0)
    else:
        loadvideo(f"video/{current_video}.mp4")
    return stream_mode


@ application.route('/next', methods=['POST'])
def get_current_video():
    global current_video
    global pause
    current_video = request.form['status']
    loadvideo(f"video/{current_video}.mp4")
    return current_video


@ application.route('/set_current_model', methods=['POST'])
def set_current_model():
    global current_model
    current_model = request.form['status']
    set_model(current_model)
    return current_model


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80)
