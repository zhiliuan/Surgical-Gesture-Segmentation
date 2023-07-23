import os
import numpy as np
import re
import math
import cv2
import os
import numpy as np
import PIL
from PIL import Image




def change_size(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom
    pre1_picture = image[left:left + width, bottom:bottom + height]
    # print(pre1_picture.shape)
    return pre1_picture


def video_frame():
    for video_name in VideoNames:
        frame_num = 0
        if not os.path.exists(frame_path + str(video_name[0:-4])):
            os.mkdir(frame_path + str(video_name[0:-4]))

        cap = cv2.VideoCapture(source_path + str(video_name))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_save_path = frame_path + str(video_name[0:-4]) + '/' + str(frame_num) + ".jpg"
            dim = (int(frame.shape[1] / frame.shape[0] * 300), 300)
            frame = cv2.resize(frame, dim)
            frame = change_size(frame)
            img_result = cv2.resize(frame, (640, 480))
            # print(img_result.shape)
            # print(img_result.dtype)

            img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
            img_result = PIL.Image.fromarray(img_result)
            # print(img_result.mode)

            img_result = np.ascontiguousarray(img_result)
            cv2.imwrite(img_save_path, img_result)
            # print(img_save_path)
            frame_num = frame_num + 1
            cv2.waitKey(1)
        print("finish with ", video_name)
    cap.release()
    cv2.destroyAllWindows()
    print("Cut Done")


def truth_seg_label(transcript_file):
    annotation = {}
    index = 0
    start = []
    end = []
    label = []
    start_index = open(transcript_file)
    for lines in start_index.readlines():
        column = lines.split(" ")
        column = [i for i in column if (len(str(i)) != 0)]
        annotation[index] = [column[0], column[1], column[2]]  # start,end,label
        start.append(int(column[0]))
        end.append(int(column[1]))
        label.append(column[2])
        index += 1
    return start, end, label


def get_match_framefile(framefile_name):
    match_frame_file = []
    for f in framefile_name:  # Suturing_B001_capture1, Suturing_B001_capture2
        aaa = sn[0:-4]  # eliminate suffix
        if re.search(sn[0:-4], f):
            match_frame_file.append(f)
            if len(match_frame_file) == 2:
                break
    return match_frame_file


if __name__ == "__main__":
    homepath = os.getcwd() + "/"
    source_path = homepath + "all videos/"  # original path
    VideoNames = os.listdir(source_path)
    frame_path = homepath + "all frames/" # save path
    transcript_path = homepath + "all transcripts/"
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    video_frame()
    all_frame_name = os.listdir(frame_path)

    session = os.listdir(transcript_path)  # Suturing_B001.txt
    session.sort()
    data_train_dict = {}
    data_path = []
    data_train_npz = []
    for sn in session:  # Suturing_B001.txt

        transcript_file = transcript_path + sn
        start, end, label = truth_seg_label(transcript_file)
        match_frame_file = get_match_framefile(all_frame_name)

        for capture in range(len(match_frame_file)):
            i = 0
            step = 1
            length = 30
            one_frame_path = frame_path + match_frame_file[capture] # Suturing_B001_capture1, Suturing_B001_capture2
            img_names = os.listdir(one_frame_path)  #0.jpg 1.jpg
            img_names.sort(key=lambda x: int(x[0:-4]))
            anno_idx = 0
            for i in range(len(start)):
                start_frm = start[i]
                end_frm = end[i]
                label_gt = label[i]
                """method 1: append all images with sliding window = 30, step = 1"""
                for im in range(start_frm,end_frm,step):
                    if end_frm - start_frm >= length:
                        idx = im
                    elif end_frm - start_frm < length:
                        idx = im -( length-(end_frm - start_frm))

                    if idx+length <= end_frm:
                        data_p = img_names[idx:idx+length]
                        for img in data_p:
                            data_path.append(frame_path + match_frame_file[capture] + '/' + img)
                            if len(data_path) == length:
                                data_train_dict['session'] = sn
                                data_train_dict['label'] = label_gt
                                data_train_dict['record'] = int(capture)
                                data_train_dict['data'] = data_path
                                new_dic = data_train_dict.copy()
                                data_train_npz.append(new_dic)
                                data_path = []
            if capture == 0:
                np.savez(sn[0:-4]+'.npz', data_train_npz)
                data_train_npz = []
            else:
                np.savez(sn[0:-4]+'_cap2'+'.npz', data_train_npz)
                data_train_npz = []
    # print("finish with npz with step 1", session)

