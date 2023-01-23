import cv2
import os
import datetime
from pathlib import Path
import numpy as np
import mediapipe as mp
from pathlib import Path
import json

def videos_to_frames(video_url_path):
    if os.path.isfile(video_url_path) == True:
        # Read the video from specified path
        cap = cv2.VideoCapture(video_url_path)
        #Create unique directory
        basename = os.path.basename(video_url_path)
        foldername = os.path.splitext(basename)[0]
        unique_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        new_path = './media/videos/screenshots/' + str(foldername) + '-' + str(unique_id) + '/'

        #Creates new path for screencaps and annotated images
        os.makedirs(new_path)
        os.makedirs(new_path+"annotated/")

        currentframe = 0
        while (True):
            ret, frame = cap.read()

            if ret:
                date = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                file_name = str(new_path) + str(date) + '.jpg'
                
                cv2.imwrite(file_name, frame)
                currentframe += 30
                cap.set(cv2.CAP_PROP_POS_FRAMES, currentframe)
            else:
                break
    else:
        print("Video path is not correct")

    cap.release()
    cv2.destroyAllWindows()
    return new_path

def sort_frames_to_list(new_path):
    screencaps_path = Path(new_path).glob("*.jpg")
    screencaps_list = ["./"+str(i) for i in screencaps_path]
    screencaps_list_sorted = sorted(screencaps_list)
    return screencaps_list_sorted

def annotate_frame(static_screencaps):
    
    BG_COLOR = (192, 192, 192) # gray

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5) as pose:
    
        for idx, file in enumerate(static_screencaps):
            path_dir = os.path.dirname(file)
            base_object_name = os.path.splitext(os.path.basename(file))[0]
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape

            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks is None:
                print("no pose found")

            else:
                nose_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
                nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
                l_shldr_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
                l_shldr_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
                r_shldr_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
                r_shldr_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
                l_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
                l_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
                r_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
                r_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height

                results_output = []

                with open(str(path_dir)+'/annotated/'+str(idx)+'-'+str(base_object_name)+ '.json', 'w') as f:
                    data = {"nose_x":str(nose_x),"nose_y":str(nose_y),"l_shldr_x":str(l_shldr_x),"l_shldr_y":str(l_shldr_y),"r_shldr_x":str(r_shldr_x),"r_shldr_y":str(r_shldr_y),
                        "l_hip_x":str(l_hip_x),"l_hip_y":str(l_hip_y),"r_hip_x":str(r_hip_x),"r_hip_y":str(r_hip_y) }

                    results_output.append(data)
                    json.dump(results_output, f)
            
                annotated_image = image.copy()

                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
                annotated_image = np.where(condition, annotated_image, bg_image)
                
                # Draw pose landmarks on the image.
                mp_drawing.draw_landmarks(annotated_image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imwrite(str(path_dir)+'/annotated/'+str(idx)+'-'+str(base_object_name)+ '.png', annotated_image)
    return print("analysis done!")

def annotate_choreography(url_path):
    frames_path = videos_to_frames(url_path)
    frames_list = sort_frames_to_list(frames_path)
    annotate_frame(frames_list)
    print("see the results at ",frames_path)


import librosa
import librosa.display
import matplotlib.pyplot as plt

def get_tempo(video_url_path):
    y, sr = librosa.load(video_url_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beat_frames

def get_beat_times(video_url_path):
    y, sr = librosa.load(video_url_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times