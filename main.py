import time
import logging
import os, sys, subprocess, pkg_resources
import datetime
import cv2



def install_python_libraries(package):
    args_list = [sys.executable, "-m", "pip", "install", package]

    subprocess.call(args_list)


def check_and_install_libraries():
    for package in ['pyrealsense2', 'matplotlib', 'tensorflow', 'opencv-python', 'numpy', 'mediapipe']:
        try:
            dist = pkg_resources.get_distribution(package)
            print('{} ({}) is installed'.format(dist.key, dist.version))
        except pkg_resources.DistributionNotFound:
            print('{} is NOT installed. Installing it'.format(package))
            install_python_libraries(package)


check_and_install_libraries()


import cv2
import mediapipe as mp

from realsense_camera import *
import pyrealsense2 as rs2
import os
import csv
class PoseDetector:

    # def __init__(self, mode=False, upBody=False, test = True, smooth=True, modelComplex = 1, detectionCon=0.8, trackCon=0.5):
    #     self.mode = mode
    #     self.upBody = upBody
    #     self.smooth = smooth
    #     self.detectionCon = detectionCon
    #     self.trackCon = trackCon
    #     self.modelComplex = modelComplex
    #     self.mpDraw = mp.solutions.drawing_utils
    #     self.mpPose = mp.solutions.pose
    #     self.mpTest = test
    #     self.pose = self.mpPose.Pose(self, self.mode, self.modelComplex, self.upBody, self.mpTest, self.smooth, self.detectionCon, self.trackCon)
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():

    object_to_track = [24, 0, 11, 12,14,16] #24
    rs = RealsenseCamera()
    detector = PoseDetector()

    cTime = 0
    pTime = 0

    recording = 0
    if os.path.exists("data.csv"):
        os.remove("data.csv")
        print("The file has been deleted successfully")
    else:
        print("The file does not exist!")

    f = open('../data.csv', 'w', newline='')
    writer = csv.writer(f)
    header = ['rhipx', 'rhipy', 'rhipz', 'neckx', 'necky', 'neckz','lshoulderx','lshouldery','lshoulderz','rshoulderx','rshouldery','rshoulderz', 'relbowx','relbowy','relbowz','rwristx','rwristy','rwristz']
    writer.writerow(header)
    while True:

        # ----------------Capture Camera Frame-----------------
        ret, color_image, depth_image, depth_colormap = rs.get_frame_stream()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = color_image.copy()
        frame = detector.findPose(frame)






        depth_frame = rs.get_frames().get_depth_frame()
        color_frame = rs.get_frames().get_color_frame()


        lmList = detector.getPosition(frame)

        if len(lmList) != 0:
            row = []
            for objet in object_to_track:

                _, x, y = lmList[objet]

                #print(y)
                if(x < 640 and y< 480 and x>=0 and y>=0):
                    #print(x)
                    #print(y)
                    depth = depth_frame.get_distance(x, y)
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    # print(depth)
                    depth_to_object = depth_image[y, x]
                    depth_point = rs2.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_to_object / 1000)  #
                    #if (depth_point[0] == 0 and depth_point[2] == 0):
                        #print('DEBUG: ', lmList[objet])
                    #else:
                       #print(lmList[objet])
                    # print(lmList[objet])
                    text = "%.5lf, %.5lf, %.5lf\n" % (
                        depth_point[0], depth_point[1], depth_point[2])
                    # row.append(-depth_point[0])
                    # row.append(-depth_point[2])
                    # row.append(-depth_point[1])
                    if objet != 0:
                        row.append(depth_point[0])
                        row.append(depth_point[2])
                        row.append(-depth_point[1])
                    else:
                        row.append(0)
                        row.append(0)
                        row.append(0)
                    # print(text)
                    # print(depth_to_object)
                    # print(depth)
                    # print(text)


            if recording ==1 and row!= []:
                writer.writerow(row)
            #text = "Depth: {} cm".format(depth) + " ".format(depth_to_object)
            text = "%.5lf, %.5lf" % (depth, depth_to_object)
            #print(text)
            cv2.putText(frame, text, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            #rs.__init__()
            #depth_frame1 = rs.get_frames().get_depth_frame()
            #print(depth_frame1.profile)
            #print(depth_frame)
            #print(rs2.motion_stream_profile.as_motion_stream_profile().get_extrinsics_to(depth_frame1.profile))
            #print(depth_frame.profile.as_video_stream_profile().get_extrinsics_to(depth_frame1.profile))

        else:
            print("object did not detected")



        # calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Draw framerate in corner of frame
        cv2.putText(frame, 'FPS: {0:.2f}'.format(fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        # # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Frame RGB', frame)
        cv2.imshow('depth image colormap', depth_colormap)
        # cv2.imshow("depth_image", depth_image)

        if cv2.waitKey(1) == ord('a'):
            recording = 1
            print("Start Writing In")
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    f.close()
    rs.release()

# Clean up
cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
