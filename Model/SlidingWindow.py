import cv2
import time
import os
import math
from tf.keras.models import load_model

def extract_and_predict_from_video(input_directory, output_directory, exercises):
    # Check whether the target/output directory exist. If not create a new one.
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    except OSError:
        print ('Error: Creating target directory of data')

    # Get trained model 
    pose_model = load_model('./full_approach_training_history/saved_models/my_model.h5')
    activity_model = load_model('./full_approach_training_history/saved_models/my_model.h5')

    # Log the time
    time_start = time.time()

    # Start capturing the feed
    cap = cv2.VideoCapture(input_directory)

    # Start extracting video snippets and predicting
    while cap.isOpened():

        # Frame Id
        frame_id = cap.get(1)  # current frame number

        # Extract the frame
        ret, frame = cap.read()
        
        # Break if no more images are left 
        if not ret:
            # Log the end-time 
            time_end = time.time()

            # Release the capture
            cap.release()
            cv2.destroyAllWindows()

            # Print stats
            print("Done! Extracted %(num_frames)d frames and create %(num_frames)d mirror frames." %{'num_frames': frame_id + 1})
            print("Extracting and flipping each images took %d seconds for conversion." % (time_end-time_start))
            print("=====================================================================")
            break
        
        # Scale images
        scaled_frame = resize_scale_individually(frame, 801, 450)

    if cv2.waitKey(1) == 27:
        break

    data = loadPoses('./PoseEstimation_Data')

def resize_scale_individually(image, width = 1920, height = 1080):
    # get traget scale/dimension
    dim = (width, height)
 
    # resize image
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)  

if __name__=="__main__":
    # Get all ".mp4" files in the input directory 
    dir = './Test_Videos/'
    files = [fname for fname in dir if fname.endswith('.mp4')]
    
    for VideoFile in files:
        # Video file name with out type extension
        VideoFileName = VideoFile.split('.')[0]

        # Extract long left dives
        input_directory = dir
        output_directory = dir
        outputfile = VideoFileName

        # Extracte and store each frame from the video snippet. In addition generate a mirror version of all frames.
        extract_and_predict_from_video(input_directory, output_directory, outputfile)

    
        

