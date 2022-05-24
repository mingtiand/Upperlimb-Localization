import time
import logging
import os, sys, subprocess, pkg_resources, csv, math
import datetime
import cv2
import matplotlib.pyplot as plt

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
#from inversekinematics import *
import pyrealsense2 as rs2
import pupil_apriltags as apriltag

class PoseDetector:

    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        ###
        POSE_CONNECTIONS = frozenset([(11, 12), (11, 13),(13, 15),
                                      (12, 14), (14, 16), (11, 23), (12, 24),(23,24)])
        ###
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, POSE_CONNECTIONS) ###self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape

                #cx, cy , vis= int(lm.x * w), int(lm.y * h), float(lm.visibility)
                #lmList.append([id, cx, cy, vis])
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                #if draw:
                    #cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


filt0_1,filt0_2,filt0_3,filt0_4 = 0,0,0,0
ang0_1,ang0_2,ang0_3,ang0_4 = 0,0,0,0
def lowpass(ang1,ang2,ang3,ang4):
    global ang0_1,ang0_2,ang0_3,ang0_4
    global filt0_1,filt0_2,filt0_3,filt0_4
    filt1 = 0.683 * filt0_1 + 0.159 * ang1 + 0.159 * ang0_1
    filt2 = 0.683 * filt0_2 + 0.159 * ang2 + 0.159 * ang0_2
    filt3 = 0.683 * filt0_3 + 0.159 * ang3 + 0.159 * ang0_3
    filt4 = 0.683 * filt0_4 + 0.159 * ang4 + 0.159 * ang0_4
    [ang0_1, ang0_2, ang0_3, ang0_4] = [ang1,ang2,ang3,ang4]
    [filt0_1,filt0_2, filt0_3, filt0_4]  = [filt1, filt2, filt3, filt4]
    return filt1, filt2, filt3, filt4

def getKinematic1(file):
    P_hip = [file[0], file[1] + 0.1, file[2]]
    P_ne = [file[3], file[4], file[5]]
    P_sh_l = [file[6], file[7], file[8]]
    P_shoulder = [file[9], file[10], file[11]]
    P_elbow = [file[12], file[13], file[14]]
    P_wrist = [file[15], file[16], file[17]]

    P_hip = [float(P_hip[i]) for i in range(len(P_hip))]
    P_ne = [float(P_ne[i]) for i in range(len(P_ne))]
    P_sh_l = [float(P_sh_l[i]) for i in range(len(P_sh_l))]
    P_shoulder = [float(P_shoulder[i]) for i in range(len(P_shoulder))]
    P_elbow = [float(P_elbow[i]) for i in range(len(P_elbow))]
    P_wrist = [float(P_wrist[i]) for i in range(len(P_wrist))]

    P_se = np.asarray([float(P_elbow[i]) - float(P_shoulder[i]) for i in range(len(P_shoulder))])
    P_sw = np.asarray([float(P_wrist[i]) - float(P_shoulder[i]) for i in range(len(P_shoulder))])

    P_lr = np.asarray([float(P_shoulder[i]) - float(P_sh_l[i]) for i in range(len(P_shoulder))])
    P_ew = np.asarray([float(P_wrist[i]) - float(P_elbow[i]) for i in range(len(P_shoulder))])
    B = np.asarray([float(P_wrist[i]) - float(P_elbow[i]) for i in range(len(P_shoulder))])
    A = np.asarray([float(P_shoulder[i]) - float(P_elbow[i]) for i in range(len(P_shoulder))])

    x_axis = P_lr / np.sqrt(P_lr.dot(P_lr))
    y_axis = np.cross(x_axis, np.asarray([P_hip[i] - P_shoulder[i] for i in range(len(P_shoulder))]))
    y_axis = y_axis / np.sqrt(y_axis.dot(y_axis))
    z_axis = np.cross(x_axis, y_axis);
    z_axis = z_axis / np.sqrt(z_axis.dot(z_axis))

    TT = np.array([x_axis, y_axis, z_axis])

    P_se2 = np.dot(TT, P_se)

    beta_shoulder = math.atan2(P_se2[2], math.sqrt(P_se2[1] ** 2 + P_se2[0] ** 2)) + math.pi / 2;
    if P_se2[1] < 0:
        beta_shoulder = -beta_shoulder

    elbow_ang = math.acos(np.dot(A / np.linalg.norm(A), B / np.linalg.norm(B)))
    elbow_d = elbow_ang * 180 / math.pi

    beta_d = beta_shoulder * 180 / math.pi

    if abs(beta_d) < 1:
        P_sw2 = np.dot(TT, P_sw)
        alpha_shoulder = math.atan2(P_sw2[1], P_sw2[0]) - math.pi / 2
        gamma_shoulder = 0
    else:
        alpha_shoulder = math.atan2(P_se2[1], P_se2[0]) - math.pi / 2

        P_wrist = np.dot(TT, P_wrist)
        P_elbow = np.dot(TT, P_elbow)
        len_f = np.linalg.norm(B)

        if beta_shoulder == 0 or elbow_ang == 0:
            gamma_shoulder = 0

            gamma_d = 0

        else:
            a = (math.cos(beta_shoulder) * math.cos(elbow_ang) - ((P_wrist[2] - P_elbow[2]) / len_f)) / math.sin(
                beta_shoulder) / math.sin(elbow_ang)

            if a > 1 or a < -1:
                gamma_shoulder = math.pi / 2
            else:
                gamma_shoulder = math.acos(a) - math.pi

            P_ew = np.dot(TT, P_ew)
            ang3 = math.atan2(P_ew[1], P_ew[0])

            if ang3 > (alpha_shoulder + math.pi / 2):
                gamma_shoulder = -gamma_shoulder

    alpha_d = alpha_shoulder * 180 / math.pi
    gamma_d = gamma_shoulder * 180 / math.pi
    if alpha_d < -180:
        alpha_d = alpha_d + 360
    alpha_d, beta_d, gamma_d, elbow_d= lowpass(alpha_d, beta_d, gamma_d, elbow_d)

    row = [alpha_d, beta_d, gamma_d, elbow_d]

    return row


def main():
    at_detector = apriltag.Detector(families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
    object_to_track = [23, 24, 11, 12,14,16] #24
    rs = RealsenseCamera()
    #ik = inversekinematic()
    detector = PoseDetector()

    cTime = 0
    pTime = 0

    recording = 0
    dataID = "5_1"
    fileName = "data" + dataID + ".csv"

    if os.path.exists(fileName):
        os.remove(fileName)
        print("The file has been deleted successfully")
    else:
        print("The file does not exist!")

    f = open('../'+fileName, 'w', newline='')
    writer = csv.writer(f)
    header = ['dataTime','lhipx', 'lhipy', 'lhipz', 'rhipx','rhipy', 'rhipz',\
             'lshoulderx','lshouldery','lshoulderz','rshoulderx','rshouldery','rshoulderz', \
              'relbowx','relbowy','relbowz','rwristx','rwristy','rwristz', \
              'rttagx', 'rttagy', 'rttagz','lttagx', 'lttagy', 'lttagz', 'rbtagx', 'rbtagy', 'rbtagz',\
              'vislhip','visrhip','vislshoulder','visrshoulder','visrelbow','visrwrist']
    writer.writerow(header)

    if sys.argv[1] == "0":
        print("No Visualization")
    else:
        updated_x = np.linspace(0, 1, 100) #0 0 0 90
        #updated_y = np.linspace(-180, 180, 100)
        if sys.argv[1] == "1" or sys.argv[1] == "2" or sys.argv[1] == "3":
            updated_y = np.zeros(100)
        elif sys.argv[1] == "4":
            updated_y = np.ones(100)*90

        plt.ion()
        figure, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylim(-180, 180)
        line1, = ax.plot(updated_x, updated_y)
        plt.title("DoF real-time", fontsize=25)
        plt.xlabel("Time ", fontsize=18)
        plt.ylabel("Degree ($\circ$)", fontsize=18)

    n = 0
    while True:


        # ----------------Capture Camera Frame-----------------
        ret, color_image, depth_image, depth_colormap = rs.get_frame_stream()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = color_image.copy()
        frame = detector.findPose(frame)

        depth_frame = rs.get_frames().get_depth_frame()
        color_frame = rs.get_frames().get_color_frame()


        lmList = detector.getPosition(frame)



        if (n==0):
            n = n+1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=[1.93 * 640, 1.93 * 480, 320, 240], #1.93
                                      tag_size=0.1)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=[1.93 * 640, 1.93 * 480, 320, 240],
                                      tag_size=0.1)
        if(len(tags) != 0):
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            rbtag = tags[0].corners[1]

            depth_to_object = depth_image[int(rbtag[1]), int(rbtag[0])]
            depth_point = rs2.rs2_deproject_pixel_to_point(depth_intrin, rbtag, depth_to_object / 1000)
            rbtag = [depth_point[0],depth_point[2],-depth_point[1]]

            rttag = tags[0].corners[2]
            depth_to_object = depth_image[int(rttag[1]), int(rttag[0])]
            depth_point = rs2.rs2_deproject_pixel_to_point(depth_intrin, rttag, depth_to_object / 1000)
            rttag = [depth_point[0],depth_point[2],-depth_point[1]]

            lttag = tags[0].corners[3]
            depth_to_object = depth_image[int(lttag[1]), int(lttag[0])]
            depth_point = rs2.rs2_deproject_pixel_to_point(depth_intrin, lttag, depth_to_object / 1000)
            lttag = [depth_point[0],depth_point[2],-depth_point[1]]
            for tag in tags:
                cv2.circle(frame, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
                cv2.circle(frame, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2)  # right-top
                cv2.circle(frame, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2)  # right-bottom
                cv2.circle(frame, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom

        else:
            rbtag = [0,0,0]
            rttag = [0,0,0]
            lttag = [0,0,0]

        if len(lmList) != 0:
            row = []
            #visibility = []
            #row.append(str(datetime.datetime.now()))
            for objet in object_to_track:

                #_, x, y, vis = lmList[objet]
                _, x, y = lmList[objet]
                if(x < 640 and y< 480 and x>=0 and y>=0):
                    #print(x)
                    #print(y)

                    depth = depth_frame.get_distance(x, y)
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    # print(depth)
                    depth_to_object = depth_image[y, x]
                    # if objet == 23:
                    #     print('23: ' + str(vis))
                    #     print(depth_to_object/1000)
                    # if objet == 24:
                    #     print('24: ' + str(vis))
                    #     print(depth_to_object/1000)
                    #print(depth_intrin)
                    depth_point = rs2.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_to_object / 1000)  #
                    #if (depth_point[0] == 0 and depth_point[2] == 0):
                        #print('DEBUG: ', lmList[objet])
                    #else:
                       #print(lmList[objet])
                    # print(lmList[objet])
                    #text = "%.5lf, %.5lf, %.5lf\n" % (
                    #    depth_point[0], depth_point[1], depth_point[2])
                    # row.append(depth_point[0])
                    # row.append(depth_point[2])
                    # row.append(-depth_point[1])

                    row.append(depth_point[0])
                    row.append(depth_point[2])
                    row.append(-depth_point[1])
                    #visibility.append(vis)

                    # print(text)
                    # print(depth_to_object)
                    # print(depth)
                    # print(text)
                else:
                    row.append(0)
                    row.append(0)
                    row.append(0)
                    #visibility.append(vis)
            #angle = ik.getKinematic(row)
            angle = getKinematic1(row)
            #print(angle)
            #print(angle)
            #updated_y = np.append(updated_y,angle[0])
            #updated_x = np.append(updated_x,updated_x[len(updated_x)-1]+1)
            if sys.argv[1] == '0':
                pass
            else:
                updated_y =np.delete(updated_y,0)
                updated_y = np.append(updated_y,angle[int(sys.argv[1])-1])
                line1.set_ydata(updated_y)
                figure.canvas.draw()
                figure.canvas.flush_events()



            text = ""
            cv2.putText(frame, text, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            #row.extend(rttag)
            #row.extend(lttag)
            #row.extend(rbtag)
            #row.extend(visibility)

            #if recording ==1 and row!= []:
                #writer.writerow(row)

            #text = "Depth: {} cm".format(depth) + " ".format(depth_to_object)
            #text = "%.5lf, %.5lf" % (depth, depth_to_object)

            #print(text)

            #rs.__init__()qqq
            #depth_frame1 = rs.get_frames().get_depth_frame()
            #print(depth_frame1.profile)
            #print(depth_frame)
            #print(rs.pipe().get_stream(rs.stream()).as_pose_stream_profile().get_extrinsics_to(depth_frame.profile))
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
        #cv2.imshow('depth image colormap', depth_colormap)
        # cv2.imshow("depth_image", depth_image)
        #if cv2.waitKey(1) == ord('a'):
            #recording = 1
            #print("Start Writing In")
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    f.close()
    rs.release()


# Clean up
cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

