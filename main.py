from model_constants import FRAME_WINDOW
from ultralytics import YOLO
from flask import Flask, request, jsonify
from threading import Thread
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv



def process_frames(frames):
    results = {}
    frame_nmr = -1
    mot_tracker = Sort()

    vehicles = [2, 3, 5, 7]

    for frame in frames:
        frame_nmr += 1
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

    # write results
    write_csv(results, './test.csv')
    return results


app = Flask(__name__)
# Frame buffer
frame_buffer = []

@app.route('/detect_license_plate', methods=['POST'])
def detect_license_plate():
    image_file = request.files['image'].read()
    np_image = np.frombuffer(image_file, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Add the frame to the buffer
    frame_buffer.append(frame)

    response_data = []

    # Check if buffer has 5 frames
    if len(frame_buffer) == FRAME_WINDOW:
        # Use the process_frames function
        results = process_frames(frame_buffer)

        # Clear the buffer after processing
        frame_buffer.clear()

        # Process the results to prepare the response for each frame
        for idx in range(FRAME_WINDOW):
            frame_data = results.get(idx, {})
            for car_id, data in frame_data.items():
                license_plate_data = data.get('license_plate', {})
                license_plate_text = license_plate_data.get('text', None)
                license_plate_text_score = license_plate_data.get('text_score', None)

                if license_plate_text:
                    response_data.append({
                        'frame_number': idx,
                        'car_id': car_id,
                        'license_plate_text': license_plate_text,
                        'probability': license_plate_text_score
                    })

    return jsonify(response_data)



def listen_continuously():
    while True:
        print("waiting...")
        time.sleep(2)

if __name__ == '__main__':
    # Load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    
    # Start the continuous listening function in a separate thread
    listen_thread = Thread(target=listen_continuously)
    listen_thread.start()

    # Run Flask server
    app.run(debug=True, port=5000)