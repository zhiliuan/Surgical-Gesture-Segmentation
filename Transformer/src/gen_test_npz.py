import os
import numpy as np
import re
import math


transcript_paths = ['D:/J/Suturing/result file/final segment points/','D:/J/Needle_Passing/result file/all segment points/',
                    'D:/J/Knot_Tying/result file/final segment points/']
task_names = ['Suturing','Needle Passing','Knot Tying']
frame_paths = ['D:/J/Transformer_mix/all_frames/Suturing/','D:/J/Transformer_mix/all_frames/Needle Passing/','D:/J/Transformer_mix/all_frames/Knot Tying/']

for i in range(len(transcript_paths)):
    transcript_path = transcript_paths[i]
    session = os.listdir(transcript_path)  # Suturing_B001.txt
    session.sort()
## change the frame path to where the images are.
## images are orginally located at ../suturing/frame/
    colab_frame_path = "/content/Transformer_mix/all_frames/"+task_names[i]+'/'
    frame_path = frame_paths[i]

    data_train_dict = {}
    data_path = []
    data_train_npz = []
    frame_name = os.listdir(frame_path)
    for sn in session:  # Suturing_B001.txt
        index = 0
        start = []
        end = []
        label = []
        i = 0
        start_index = open(transcript_path + sn)
        for lines in start_index.readlines():
            column = lines.split(" ")
            start.append(int(column[0]))
            end.append(int(column[0]) - 1)
            label.append(i)
            i += 1
        end = end[1:]
        start = start[:-1]

        match_frame_file = []
        for f in frame_name:  # Suturing_B001_capture1, Suturing_B001_capture2
            aaa = sn[0:-4]
            if re.search(sn[0:-4], f):
                match_frame_file.append(f)
                if len(match_frame_file) == 2:
                    break

        for capture in range(len(match_frame_file)):
            i = 0
            step = 30
            length = 30
            one_frame_path = frame_path + match_frame_file[capture]  # Suturing_B001_capture1, Suturing_B001_capture2
            img_names = os.listdir(one_frame_path)  # 0.jpg 1.jpg
            img_names.sort(key=lambda x: int(x[0:-4]))
            anno_idx = 0

            for i in range(len(start)):
                start_frm = start[i]
                end_frm = end[i]
                """method 1: append all images with sliding window = 30, step = 30"""

                if end_frm - start_frm >= length:
                    data_p = img_names[start_frm: start_frm + length]
                elif end_frm - start_frm < length:
                    copy_ammount = length-(end_frm - start_frm) -1
                    data_p = [img_names[start_frm] for _ in range(copy_ammount)]+img_names[start_frm:end_frm+1]
                for img in data_p:
                    data_path.append(colab_frame_path + match_frame_file[capture] + '/' + img)
                    if len(data_path) == length:
                        label_gt = start_frm
                        data_train_dict['session'] = sn
                        data_train_dict['label'] = label_gt
                        data_train_dict['record'] = int(capture)
                        data_train_dict['data'] = data_path
                        new_dic = data_train_dict.copy()
                        data_train_npz.append(new_dic)
                        data_path = []
            if capture == 0:
                np.savez("predict_" + sn[0:-4] + '.npz', data_train_npz)
                data_train_npz = []
            else:
                np.savez("predict_" + sn[0:-4] + '_cap2' + '.npz', data_train_npz)
                data_train_npz = []

    print("finish")
