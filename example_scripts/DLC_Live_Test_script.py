from dlclive import DLCLive, Processor
import cv2

dlc_proc = Processor()
model_path = "C:/Users/Legos/Documents/PhD/Blender/OmniTrax/OmniTrax_WIP/DLC_multi_ant_test_label_resnet_50_iteration-0_shuffle-1"
dlc_live = DLCLive(model_path, processor=dlc_proc, dynamic=(False, 0.1, 10))

video_path = "C:/Users/Legos/Documents/PhD/Blender/OmniTrax/OmniTrax_WIP/data/single_ant_1080p.mp4"
cap = cv2.VideoCapture(video_path)

max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0
print(max_frames)

plot_IDs = [0, 1, 2, 3, 4, 5, 6]  # ,8,9,15,16,22,23,29,30,36,37,43,44]

while cap.isOpened():
    frame_num += 1
    if frame_num == max_frames:
        break

    ret, frame = cap.read()
    try:
        if ret == True:
            if frame_num == 1:
                dlc_live.init_inference(frame)

            pose = dlc_live.get_pose(frame)
            print(pose.shape)

            for p, point in enumerate(pose):
                if p in plot_IDs:
                    frame = cv2.circle(
                        frame,
                        (int(point[0]), int(point[1])),
                        5,
                        (int(255 * point[2]), int(100 * point[2]), 200),
                        -1,
                    )

            cv2.imshow("DLC-Live_output", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except:
        print("Yeah, that's enough for now.")

cv2.destroyAllWindows()
print("Read all frames")
