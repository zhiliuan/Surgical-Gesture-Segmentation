from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xlsxwriter as xw
import pandas as pd
import xlrd

preseg_path = 'result file/potensial segment points.xls'
preseg_workbook = xlrd.open_workbook(preseg_path)
sheet_names= preseg_workbook.sheet_names()

label_file = "./result file/all segment points.xlsx" #工作簿名字
workbook = xw.Workbook(label_file)  # 创建工作簿

for sheet_name in sheet_names:
    trans_preseg = []
    rot_preseg = []
    var_preseg = []
    acc_preseg = []
    sheet = preseg_workbook.sheet_by_name(sheet_name)
    # rows = sheet.nrows # 获取行数
    cols = sheet.ncols # 获取列数，尽管没用到
    for c in [4,5,6,7,13,14]: #cwt
        data = sheet.col_values(c)[1:-1]
        data = [i for i in data if i != ""]
        trans_preseg.extend(data)
    trans_preseg.sort()
    trans_preseg = np.array(trans_preseg).reshape(-1, 1)
    for r in [9,10,11,12,15,16,19,20]:
        datar = sheet.col_values(r)[1:-1]
        datar = [i for i in datar if i != ""]
        var_preseg.extend(datar)
    var_preseg.sort()
    var_preseg = np.array(var_preseg).reshape(-1, 1)

    all_character = np.vstack((trans_preseg,var_preseg)).flatten()
    all_character.sort()
    all_character = np.array(all_character).reshape(-1, 1)

    db3= cluster.DBSCAN(eps = 15, min_samples = 3)
    db3.fit(all_character)
    all_character_label = db3.labels_
    all_character_core = db3.core_sample_indices_
    print("all_character",all_character_label)

    db1= cluster.DBSCAN(eps = 15, min_samples = 2)
    db1.fit(trans_preseg)
    trans_rot_label = db1.labels_
    trans_rot_core = db1.core_sample_indices_
    print("trans_rot_label",trans_rot_label)

    # db2= cluster.DBSCAN(eps = 6, min_samples = 2,  algorithm='ball_tree',leaf_size=10)
    db2= cluster.DBSCAN(eps =15 , min_samples = 3)
    db2.fit(var_preseg)
    var_label = db2.labels_
    var_core = db2.core_sample_indices_
    print("var_label",var_label)

    traindatas = [trans_preseg, var_preseg, all_character]
    all_label = [trans_rot_label,var_label,all_character_label]

    result = [[],[],[],[],[],[]]
    i = 0
    for d,ind in zip(traindatas, all_label):
        traindatas_re = d.reshape(-1)
        label_re = ind.reshape(-1)
        result[i] = traindatas_re
        result[i+1] = label_re
        i += 2
    # result = pd.DataFrame(np.vstack((traindatas_re, label_re)).T)

    # result1 = pd.DataFrame(np.vstack((result[0], result[1])).T)
    result1 = pd.DataFrame(np.vstack((trans_preseg[trans_rot_core].reshape(-1), trans_rot_label[trans_rot_core])).T)

    result1.columns = ["1", "DBSCAN label"]
    result1 = result1.drop(result1[(result1["DBSCAN label"] == -1)].index)
    result1 = result1.values

    # result2 = pd.DataFrame(np.vstack((result[2], result[3])).T)
    result2 = pd.DataFrame(np.vstack((var_preseg[var_core].reshape(-1), var_label[var_core])).T)
    result2.columns = ["2", "DBSCAN label"]
    result2 = result2.drop(result2[(result2["DBSCAN label"] == -1)].index)
    result2 = result2.values

    result3 = pd.DataFrame(np.vstack((all_character[var_core].reshape(-1), all_character_label[var_core])).T)
    result3.columns = ["3", "DBSCAN label"]
    result3 = result3.drop(result3[(result3["DBSCAN label"] == -1)].index)
    result3 = result3.values


    worksheet1 = workbook.add_worksheet(sheet_name)  # 创建子表
    worksheet1.activate()  # 激活表

    worksheet1.write_column(chr(ord('A')) + "2", result1[:,0])
    worksheet1.write_column(chr(ord('A')+1) + "2", result1[:,1])
    worksheet1.write_column(chr(ord('A')+2) + "2", result2[:,0])
    worksheet1.write_column(chr(ord('A')+3) + "2", result2[:,1])
    worksheet1.write_column(chr(ord('A')+4) + "2", result2[:,0])
    worksheet1.write_column(chr(ord('A')+5) + "2", result2[:,1])

"""plot fot result"""



workbook.close()

#[list 里面是array]
#     core_samples_mask = np.zeros_like(label, dtype=bool)  # 设置一个样本个数长度的全false向量
#     core_samples_mask[db.core_sample_indices_] = True #将核心样本部分设置为true
#     n_clusters_ = len(set(label)) - (1 if -1 in label else 0) # 获取聚类个数。（聚类结果中-1表示没有聚类为离散点）
#
#
#     unique_labels = set(label)
#     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
#
#     fig6 = plt.figure(figsize=(10, 7))
#     ax1 = fig6.add_subplot(1, 2, 1)
#     for k, col in zip(unique_labels, colors):
#         if k == -1:  # 聚类结果为-1的样本为离散点
#             # 使用黑色绘制离散点
#             col = [0, 0, 0, 1]
#         class_member_mask = (label == k)  # 将所有属于该聚类的样本位置置为true
#         xy = trans_preseg[class_member_mask & core_samples_mask]  # 将所有属于该类的核心样本取出，使用大图标绘制
#         ax1.plot(xy, 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
#
#         xy = trans_preseg[class_member_mask & ~core_samples_mask]  # 将所有属于该类的非核心样本取出，使用小图标绘制
#         ax1.plot(xy, 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# sns.despine()
# plt.show()
