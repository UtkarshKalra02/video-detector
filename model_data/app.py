from flask import Flask, render_template, Response
import cv2
import numpy as np
from Detector import Detector

app = Flask(__name__)

# Initialize the Detector with your model configuration
configPath = "model_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
modelPath = "model_data/frozen_inference_graph.pb"
classesPath = "model_data/coco.names"
detector = Detector(None, configPath, modelPath, classesPath)

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture image")
            break
        else:
            # Detect objects in the frame using the detect method
            classLabelIDs, confidences, bboxs = detector.net.detect(frame, confThreshold=0.4)

            bboxs = list(bboxs)
            confidences = list(map(float, np.array(confidences).reshape(1, -1)[0]))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = detector.classesList[classLabelID]
                    classColor = [int(c) for c in detector.colorList[classLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color=classColor, thickness=2)
                    cv2.putText(frame, displayText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, classColor, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
