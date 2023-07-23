import os
import re
import numpy as np
from collections import Counter


gt_path =['D:/J/Suturing/transcriptions/','D:/J/Needle_Passing/transcriptions/','D:/J/Knot_Tying/transcriptions/']

predict_path = 'D:/J/Transformer_mix/predict/'
predict_file = os.listdir(predict_path)

transcript_path = ['D:/J/Suturing/result file/final segment points/','D:/J/Needle_Passing/result file/all segment points/',
                    'D:/J/Knot_Tying/result file/final segment points/']
for n in range(len(gt_path)):
    session = os.listdir(transcript_path[n])  # Suturing_B001.txt
    session.sort()
    gt_file = os.listdir(gt_path[n])

    for pre_file in predict_file:
        predict_f = open(predict_path + pre_file)
        for file in gt_file:
            if re.search(file,pre_file):
                gt_f = open(gt_path[n] + file)
                break
        for sn in session:
            if re.search(sn, pre_file):
                segment_f = open(transcript_path[n] + sn)

                est_start = []
                est_end = []
                for lines in segment_f.readlines():
                    column = lines.split(" ")
                    est_start.append(int(column[0]))
                    est_end.append(int(column[0]) - 1)
                est_end = est_end[1:]
                est_start = est_start[:-1]

                frame = []
                pre_label = []
                for lines in predict_f.readlines():
                    column = lines.split(" ")
                    frame.append(int(float(column[0])))
                    pre_label.append(int(float(column[1])))

                j = 0
                est_labels = []
                est_final = []
                for idx, f in enumerate(frame):
                    if f < est_start[0] :
                        est_final.append(pre_label[idx])
                    if f == frame[-1]:
                        if len(est_labels)>1:
                            final_label = Counter(est_labels).most_common()[0][0]
                            temp = [final_label for x in range(len(est_labels))]
                            est_final.extend(temp)
                        else:
                            est_final.extend(est_labels)
                        est_labels=pre_label[idx:]
                        final_label = Counter(est_labels).most_common()[0][0]
                        temp1 = [final_label for x in range(len(est_labels))]
                        est_final.extend(temp1)
                        break
                    if est_start[j] <= f <=est_end[j]:
                        est_labels.append(pre_label[idx])
                    elif f >= est_start[j+1]:
                        if len(est_labels)>1:
                            final_label = Counter(est_labels).most_common()[0][0]
                            temp = [final_label for x in range(len(est_labels))]
                            est_final.extend(temp)
                        else:
                            est_final.extend(est_labels)
                        est_labels=[]
                        est_labels.append(pre_label[idx])
                        j+=1

                gt_start = []
                gt_end = []
                gt_label = []
                for lines in gt_f.readlines():
                    column = lines.split(" ")
                    gt_start.append(int(column[0]))
                    gt_end.append(int(column[1]))
                    gt_label.append(int(column[2][1:]))

                i = 0
                true_label = []
                for frm in frame:
                    if frm<gt_start[0]:
                        true_label.append(0)
                    elif frm>=gt_start[-1]:
                        true_label.append(gt_label[-1])
                    elif frm>=gt_start[i] and frm<=gt_end[i]:
                        true_label.append(gt_label[i])
                    elif frm>=gt_start[i+1]:
                        flag = i
                        for compare_all_idx, compare_all_gt in enumerate(gt_start[flag+1:]):
                            if compare_all_idx ==0  and frm>=compare_all_gt and i < len(gt_start) and frm<=gt_start[i+2]:
                                true_label.append(gt_label[i + 1])
                                i += 1
                                break
                            elif compare_all_idx ==0  and frm>=compare_all_gt and i==len(gt_start)-2 and frm<=gt_start[-1]:
                                true_label.append(gt_label[i + 1])
                                i += 1
                                break
                            elif compare_all_idx !=0 and frm>=compare_all_gt:
                                i += 1
                            elif compare_all_idx !=0 and frm<compare_all_gt:
                                true_label.append(gt_label[i+1])
                                i += 1
                                break

                true_label = np.array(true_label).reshape(-1,1)
                est_final = np.array(est_final).reshape(-1,1)
                frame = np.array(frame).reshape(-1,1)
                writein = np.hstack((true_label,est_final,frame))
                np.savetxt( 'classification_' + file , writein, fmt='%d', delimiter=' ')
                break


