from flask import Flask, flash, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import random
import os

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'pdf'}
noti = ""
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Name custom object
classes = open("coco.names").read().strip().split("\n")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def welcome():
    return redirect('/yolo')
 

# Route for handling the login page logic
@app.route('/yolo', methods=['GET', 'POST'])
def upload_file():
    defaults = 0
    nan = ""
    noti = ''
    mask = 0
    nomask = 0
    rate = 0.0
    noti = ''
    if request.method == 'POST':
        print("Post")
        # check if the post request has the file part
        if 'file' not in request.files:
            print("error")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print("Error")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            os.makedirs(os.path.join(app.instance_path, 'htmlfi'), exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))) 
            nan = file.filename

            # Here

            # Images path
            images_path = glob.glob(r"./upload/"+file.filename)
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
        
            # Insert here the path of your images
            random.shuffle(images_path)
            # loop through all the images
            for img_path in images_path:
                # Loading image
                img = cv2.imread(img_path)
                img = cv2.resize(img, None, fx=1, fy=1)
                height, width, channels = img.shape

                # Detecting objects
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                net.setInput(blob)
                outs = net.forward(output_layers)

            #   # Showing informations on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            # Object detected
                            
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_PLAIN
                ss = ""
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[class_ids[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
                print(mask ," " ,nomask,"\n")
                noti = ss
                cv2.imwrite("./static/result/"+ file.filename,img)

    return render_template('login.html', source=nan, df=defaults, noti=noti, rate=rate, mask=mask, nomask=nomask)
 
 
if __name__ == '__main__':
    app.run(host='localhost', port=9000, debug=True)