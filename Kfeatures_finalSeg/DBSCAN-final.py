from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xlsxwriter as xw
import pandas as pd
import xlrd
import os
import re
from bayes_opt import BayesianOptimization


def dict_insert_list(key, value, dict):
    """
    Used to maintain a dictionary of lists. For given key-value
    pair, function checks if key exists before inserting.
    """
    if key not in dict:
        dict[key] = [value, ]
    else:
        curr_list = dict[key]
        curr_list.append(value)
        dict[key] = curr_list

def metrics(list_time_clusters,start,end):
    overlap_gt = []
    for gt in range(len(start) - 1):
        overlap_gt.append(start[gt + 1] - start[gt])

    overlap_spt = []
    for spt in range(len(list_time_clusters) - 1):
        overlap_spt.append(list_time_clusters[spt + 1] - list_time_clusters[spt + 1])
    correct = 0
    for i in range(min(len(overlap_spt), len(overlap_gt))):
        if abs(list_time_clusters[i] - start[i]) + abs(list_time_clusters[i + 1] - start[i + 1]) <= 30:
            correct += 1
    if len(overlap_spt)!=0:
        recall = (correct / len(overlap_gt))
        precision = (correct / len(overlap_spt))
    else:
        recall = (correct / len(overlap_gt))
        precision = 0

    ############### single point #############
    offset = []
    for residx, res in enumerate(list_time_clusters):
        tmp = []
        for gt in start:
            tmp.append(abs(res - gt))
        offset.append(min(tmp))
    for s in offset:
        if s <= 30:
            correct += 1
            continue
    recall1 = (correct / len(list_time_clusters))
    precision1 = (correct / (len(start)-1))

    return recall, precision, recall1, precision1

def plot_res(list_time_clusters,start,end):
    # y = 10
    plt.figure(figsize=(25, 6), dpi=80)
    # 绘制条形图
    # plot ground truth
    plt.axhline(y=10, c='green',linewidth=5, xmax=0.95)
    plt.axhline(y=10.5, c='blue', linewidth=5, xmax=0.95)
    # plt.axhline(y=11, c='grey', linewidth=5, xmax=0.95)
    plt.legend(labels=["ground truth","estimated result"], loc="upper right", fontsize=15)

    ground_truthy = [10 for x in range(0,len(start))]
    plt.plot(start, ground_truthy, "ro", ms=8)

    for ver in start:
        plt.axvline(ver, color = 'green',ls='--', ymin=0.2, ymax=0.8)
        plt.text(ver, 9.5, '%d'%ver)

    smy = [10.5 for x in range(0, len(list_time_clusters))]
    plt.plot(list_time_clusters, smy, "rx", ms=12)

    for ver in list_time_clusters:
        plt.axvline(ver, ls = '--',color = 'b', ymin=0.2,ymax=0.8)
        plt.text(ver, 11, '%d'% ver,color = 'b')
    #
    # mergey = [11 for x in range(0, len(raw_data))]
    # plt.plot(raw_data, mergey, "rx", ms=12)
    plt.ylim(9,12)
    plt.xlim(0,end[-1]+500)
    plt.xlabel('frame')

def cluster_prune(label,X):
    list_of_elem = []
    for i in range(len(label)):
        list_of_elem.append((label[i], X[i]))

    list_of_elem = sorted(list_of_elem, key=lambda x: x[1][0])
    dict_time_clusters = {}
    for elem in list_of_elem:
        dict_insert_list(elem[0], elem[1], dict_time_clusters)

    list_time_clusters = []
    for ncluster in dict_time_clusters.keys():
        # get all frames in this cluster
        if ncluster!= -1:
            cluster_frames = dict_time_clusters[ncluster]
            setClusterFrames = set([elem[0] for elem in cluster_frames])
            # test if frames in cluster are representative of the test set
            min_frm = min(cluster_frames)
            max_frm = max(cluster_frames)
            leftFrame = min_frm[0]
            rightFrame = max_frm[0]
            list_time_clusters.append(leftFrame)
        else: continue

    list_time_clusters.sort()
    new_list_time_clusters = []
    for frm in range(len(list_time_clusters)-1):
        if list_time_clusters[frm+1] - list_time_clusters[frm] > 29:
           new_list_time_clusters.append(list_time_clusters[frm])
    new_list_time_clusters.append(list_time_clusters[-1])
    return new_list_time_clusters



def final_clustering(eps,num):
    # num = int(3)
    num = int(num)
    db_file_path = "result file/all segment points.xlsx"
    preseg_workbook = xlrd.open_workbook(db_file_path)
    sheet_names= preseg_workbook.sheet_names()
    TSC_file = 'D:/J/TSC_DL/result file/all/'
    tsc_names = os.listdir(TSC_file)

    label_file = "./result file/final segment points.xlsx"  # 工作簿名字
    emp = pd.DataFrame()
    emp.to_excel(label_file)
    file_open = pd.ExcelWriter(label_file)

    recall,precision = [],[]
    recall1,precision1 = [],[]
    all_files = './transcriptions/'
    all_file_name = os.listdir(all_files)
    all_file_name.sort()

    for gmm_name in all_file_name:
        # print(gmm_name)
        for tsc in tsc_names:
            aaa = gmm_name[:-4]
            if re.search(gmm_name[:-4], tsc):
                tsc_file = tsc
                break
        x = []
        tsc_files = open(TSC_file+tsc_file)
        for lines in tsc_files.readlines():
            column = lines.split(" ")
            x.append(float(column[0].rstrip('\n')))

            ## 找到对应的dp_gmm critical points文件
        for name in sheet_names:
            aaa = gmm_name[-9:-4]
            if re.search(gmm_name[-9:-4], name):
                match = name
        ## txt for final result
        f = open('./result file/all segment points/' + gmm_name , 'w')
        # 使用 ExcelFile ，通过将 xls 或者 xlsx 路径传入，生成一个实例
        trans_rot_group1 = pd.read_excel(db_file_path, engine='openpyxl', usecols=[i for i in range (2)], sheet_name= match).dropna()
        var_group1 = pd.read_excel(db_file_path, engine='openpyxl', usecols=[i for i in range (2,4)], sheet_name= match).dropna()
        trans_rot_group1.columns=["trans_index", "DBSCAN label"]
        var_group1.columns=["var_index","DBSCAN label"]
        merge_group1  = pd.read_excel(db_file_path, engine='openpyxl', usecols=[i for i in range (4,6)], sheet_name= match).dropna()
        merge_group1.columns = ["merge_index", "DBSCAN label"]
        merge_group = merge_group1["merge_index"].values.tolist()

        trans_group = trans_rot_group1.drop(trans_rot_group1[(trans_rot_group1["DBSCAN label"] == -1)].index)
        trans_group = trans_group["trans_index"].values.tolist()
        var_group = var_group1.drop(var_group1[(var_group1["DBSCAN label"] == -1)].index)
        var_group = var_group["var_index"].values.tolist()

        final_train = merge_group + x
        # final_train = merge_group
        # final_train = var_group + trans_group + x
        final_train.sort()
        all_raw = final_train
        final_train = np.array(final_train).astype(float).reshape(-1, 1)

        ############ 提取core 和对应的label ########
        db = cluster.DBSCAN(eps= eps,  min_samples= num)
        db.fit(final_train)
        label = db.labels_
        core = db.core_sample_indices_
        # print('label1',label)

        result = pd.DataFrame(np.vstack((final_train[core].reshape(-1),label[core])).T)
        result.columns=["final_index", "DBSCAN label"]
        # final_seg = result.drop(result[(result["DBSCAN label"] == -1)].index)
        df_mean = result.groupby(by='DBSCAN label').mean()
        df = result.drop(result[(result["DBSCAN label"] == -1)].index)
        segment_mean1 = df_mean['final_index'].values.tolist()

        new_list_time_clusters1 = cluster_prune(label,final_train)
        # new_list_time_clusters1.to_excel(file_open, sheet_name=match)

        """############################ plot the final result ##################################"""
        start = []
        end = []
        seg_label_path = './transcriptions/'
        transcrip_name = os.listdir(seg_label_path)
        transcrip_name.sort()
        for script in transcrip_name:
            pattern = match
            if re.search(match, script):
                match1 = script
                start_index = open(seg_label_path + script)
                for lines in start_index.readlines():
                    # print(lines)
                    column = lines.split(" ")
                    column = [i for i in column if (len(str(i)) != 0)]
                    start.append(column[0])
                    end.append(column[1])
                start.append(end[-1])
                start0_frm = int(start[0])
                end0_frm = int(end[-1])
                start = np.array(start).astype(int)
                end = np.array(end).astype(int)
                break
        if start0_frm < new_list_time_clusters1[0] and end0_frm > new_list_time_clusters1[-1]:
            writein = [start0_frm] +  new_list_time_clusters1 + [end0_frm]
        elif start0_frm > new_list_time_clusters1[0] and end0_frm > new_list_time_clusters1[-1]:
            writein =new_list_time_clusters1 + [end0_frm]
        elif start0_frm < new_list_time_clusters1[0] and end0_frm < new_list_time_clusters1[-1]:
            writein = [start0_frm] + new_list_time_clusters1
        else:
            writein =  new_list_time_clusters1
        np.savetxt(f, writein, fmt='%d', delimiter=',')
        f.close()

        """####### plot for kinematics critical points after the second DBSCAN """


        # 显示图形
        # plt.show()
        """ ###############################  metrics  ######################"""
        single_recall, single_precision, single_recall1, single_precision1 = metrics(new_list_time_clusters1, start, end)
        plot_res(new_list_time_clusters1, start, end)
        plt.savefig('./res' + match1[0:-4] + '.png')
        plt.close()
        plt.clf()
        recall.append(single_recall)
        recall1.append(single_recall1)
        precision.append(single_precision)
        precision1.append(single_precision1)


    rec_mean = np.mean(np.array(recall))
    pre_mean = np.mean(np.array(precision))
    rec1_mean = np.mean(np.array(recall1))
    pre1_mean = np.mean(np.array(precision1))
    # print('recall, precision',recall,precision)
    # print( 'rec_mean, pre_mean',  rec_mean, pre_mean)
    print('recall1',recall1)
    print('precision1 ',precision1 )
    print( 'rec1_mean, pre1_mean',  rec1_mean, pre1_mean)
    f1_mean =  2*(rec1_mean*pre1_mean)/(rec1_mean+pre1_mean)
    print('f1 score', f1_mean)

    file_open.save()
    return f1_mean

if __name__ == '__main__':
    final_clustering(30,2)
    # 0.7217   |  22.56    |  2.427   all points without DBSCAN
    ##  0.7285   |  27.09    |  2.626  drop the -1 label
    # rf_bo = BayesianOptimization(
    #     final_clustering,
    #     {'eps': (5,50),
    #      'num':(1,2)
    #      }
    # )
    # rf_bo.maximize(n_iter=100)