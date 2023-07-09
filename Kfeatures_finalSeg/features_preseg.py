import openpyxl
from sklearn import cluster
import cv2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences, peak_widths
from scipy.signal import savgol_filter
import xlsxwriter as xw
import os
import pandas as pd

from pykalman import KalmanFilter


""" To change file change in data_visualize; DP_GMM """

""" ##################################################################### """


'''#################################################### translation velocity Euclidean Distance ####################'''
def safe_concatenate(X, W):
    """
    Checks if X is None before concatenating W
    to X along specified axis (0 by default)
	"""
    if X is None:
        return W
    else:
        return np.vstack((X, W))

def KALMAN_filter(measurement1,measurement2,n):
    cor = None
    v = None
    measurement1 = np.array(measurement1, np.float32).reshape(-1, 3)
    measurement2 = np.array(measurement2, np.float32).reshape(-1, 3)
    kalman = cv2.KalmanFilter(6, n)
    kalman.transitionMatrix = np.array([ [1,0,0,1,0,0],
        [0,1,0,0,1,0],
        [0,0,1,0,0,1],
        [0,0,0,1,0,0],
       [ 0,0,0,0,1,0],
        [0,0,0,0,0,1] ], np.float32)  # 转移矩阵 A
    kalman.measurementMatrix = np.array([[1,0,0,1,0,0],
        [0,1,0,0,1,0],
        [0,0,1,0,0,1],
        [0,0,0,1,0,0],
       [ 0,0,0,0,1,0],
        [0,0,0,0,0,1] ], np.float32)  # 测量矩阵    H
    kalman.measurementNoiseCov = np.array([[1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]], np.float32) * 100  # 测量噪声        R
    kalman.processNoiseCov = np.array([[1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]], np.float32)* 1e-5  # 过程噪声 Q
    measurement = np.hstack((measurement1,measurement2))
    for i in range(len(measurement)):

        mes = np.reshape(measurement[i, :], (n, 1))
    # measurement = np.array([[np.float32(x)], [np.float32(y)]])
        c = kalman.correct(mes)
        # cor.append(c)
        current_prediction = kalman.predict()
        cor = safe_concatenate(cor, current_prediction.flatten()[0:3])
        v = safe_concatenate(v, current_prediction.flatten()[3:6])
    return cor,v



def maxminnorm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t


def eucldist_vectorized(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return np.sqrt(np.sum((coords2 - coords1) ** 2))


def two_trans_eudist(translation):
    translation = np.array(translation).astype(float).reshape(-1, 3) * 1000
    eudist = [0 for _ in range(int(len(translation) - 1))]
    for index in range(0, len(eudist)):
        eudist_res = eucldist_vectorized(translation[index], translation[index + 1])
        eudist[index] = eudist_res
    eudist = np.array(eudist).reshape(-1, 1)
    eudistance_norm = maxminnorm(eudist).reshape(-1, 1)
    return eudistance_norm


def quaternion(rotation_matrix):
    rotation_matrix = np.array(rotation_matrix)
    arr_euler = rotation_matrix.reshape(3, -1)
    r3 = R.from_matrix(arr_euler)
    qua = r3.as_quat()
    ## calculate rotation distance dist_rot ####
    # d.write(str(qua[0]) + ' ' + str(qua[1]) + ' ' + str(qua[2]) + ' ' + str(qua[3]) + '\n')
    return qua


def two_rot_eudist(rotation):
    counter = 0
    qua = []
    # dist_rot = [0 for _ in range((int(len(rotation) / 9)))]
    dist_rot = []
    eps = 1e-6
    # rotation distance
    for index in range(len(rotation)):
        while counter < len(rotation)-9:
            qua.append(quaternion(rotation[counter: counter + 9]))
            if len(qua) > 1:
                # a [0]= np.arccos(2 * np.inner(qua[i], qua[i - 1]) ** 2 - 1)
                dist_rot.append( (np.arccos(2 * np.inner(qua[-1], qua[-2]) ** 2 - 1)) )
                # dist_rot[counter - 1] = np.arccos(2 * np.inner(qua[counter], qua[counter - 1]) ** 2 - 1)
            counter = counter + 9
    dist_rot = np.array(dist_rot).reshape(-1, 1)
    dist_rot = np.nan_to_num(dist_rot)
    dist_rot_norm = maxminnorm(dist_rot).reshape(-1, 1)
    return dist_rot_norm


############################################### find prominence and corner ####################
def find_org_peaks(org_trans, org_rot):
    # org_trans = maxminnorm((-org_trans).reshape(-1, 1))
    # org_rot = maxminnorm((-org_rot).reshape(-1, 1))
    trans_index, _ = find_peaks(org_trans.flatten(), prominence=[0.05,0.5], height = 0.2) # height = 0.05
    trans_index_cwt = find_peaks_cwt(org_trans.flatten(), widths=np.arange(1,300)) # 原来（50，300）

    rot_index, _ = find_peaks(org_rot.flatten(), distance=60, prominence=0.1)
    rot_index_cwt = find_peaks_cwt(org_rot.flatten(), widths=np.arange(1, 300))# 原来（50，300）

    return trans_index, trans_index_cwt, rot_index, rot_index_cwt

def find_neg_org_peaks(org_trans, org_rot):
    # org_trans = maxminnorm((-org_trans).reshape(-1, 1))
    # org_rot = maxminnorm((-org_rot).reshape(-1, 1))
    trans_index, _ = find_peaks(org_trans.flatten(), prominence=[0.05,0.5], height = 0.8) # height = 0.05
    trans_index_cwt = find_peaks_cwt(org_trans.flatten(), widths=np.arange(1,300)) # 原来（50，300）

    rot_index, _ = find_peaks(org_rot.flatten(), distance=60, prominence=0.1)
    rot_index_cwt = find_peaks_cwt(org_rot.flatten(), widths=np.arange(1, 300))# 原来（50，300）

    return trans_index, trans_index_cwt, rot_index, rot_index_cwt

def segment_connect(start_ind, end_ind, distance, axnum):
    for segments in range(len(end_ind)):
        start_point = start_ind[segments]
        end_point = end_ind[segments]
        axnum.plot(start_point+1, distance[start_point+1], "ro", ms=5)
        axnum.plot(end_point+1, distance[end_point+1], "mo", ms=5)
        connectx = [start_point+1, end_point+1]
        connecty = [distance[start_point+1], distance[end_point+1]]
        axnum.plot(connectx, connecty, 'm--')


def new_trans(translation_orig, n, Transf):
    translation_jn = []
    translation = np.array(translation_orig).astype(float).reshape(-1, 3) * 10000
    for trans in range(0, len(translation)):
        trans_t = translation[trans] + Transf[n]
        translation_jn.append(trans_t)
    translation_jn = np.array(translation_jn).astype(float).reshape(-1, 3)
    return translation_jn


def new_rot(rotation_orig, n, RodMatrix):
    rotation_jn = []
    for rot in range(0, int(len(rotation_orig)), 9):
        rotation_l_matrix = rotation_orig[rot:rot + 9]
        rotation_l_matrix = np.array(rotation_l_matrix).astype(float).reshape(3, -1)
        R_matrix = np.dot(rotation_l_matrix, RodMatrix[n])
        sciangle_0, sciangle_1, sciangle_2 = R.from_matrix(R_matrix).as_euler('xyz')
        rotation_jn.append([sciangle_0, sciangle_1, sciangle_2])
    # print(rotation_jn)
    rotation_jn = np.array(rotation_jn)
    return rotation_jn


def generate_new_frames(num_frame, org_transdata, org_rotdata):
    Rod_matrix = [[0 for j in range(3)] for i in range(num_frame)]
    Tf = [[0] * 3 for _ in range(num_frame)]
    r = [[0] * 3 for _ in range(num_frame)]
    for new_frame in range(num_frame):
        tx = random.uniform(-20, 20)
        ty = random.uniform(-20, 20)
        tz = random.uniform(-20, 20)
        rx = random.uniform(-20, 20)
        ry = random.uniform(-20, 20)
        rz = random.uniform(-20, 20)
        r[new_frame] = np.array([rx, ry, rz])
        # Rod_matrix1 = cv2.Rodrigues(r[new_frame])[0]
        Rod_matrix[new_frame] = cv2.Rodrigues(r[new_frame])[0]
        Tf[new_frame] = [tx, ty, tz]
    Tframe = np.array(Tf).reshape(-1, 3)
    new_transframe = []
    new_rotframe = []
    for all_frames in range(num_frame):
        t = new_trans(org_transdata, all_frames, Tframe)
        new_transframe.append(t)
        new_rotframe.append(new_rot(org_rotdata, all_frames, Rod_matrix))
    new_transframe = np.array(new_transframe)
    new_rotframe = np.array(new_rotframe)
    return new_transframe, new_rotframe


def calculate_var(new_tf, new_rf):
    trans_x = new_tf[:, :, 0]
    trans_y = new_tf[:, :, 1]
    trans_z = new_tf[:, :, 2]
    rot_x = new_rf[:, :, 0]
    rot_y = new_rf[:, :, 1]
    rot_z = new_rf[:, :, 2]
    # calculate variance at each time t
    var_transx = np.array(trans_x.var(axis=0))
    var_transy = np.array(trans_y.var(axis=0))
    var_transz = np.array(trans_z.var(axis=0))
    var_trans = var_transx + var_transy + var_transz
    var_trans_norm = maxminnorm(var_trans.reshape(-1, 1))
    var_rotx = np.array(rot_x.var(axis=0))
    var_roty = np.array(rot_y.var(axis=0))
    var_rotz = np.array(rot_z.var(axis=0))
    var_rot = var_rotx + var_roty + var_rotz
    var_rot_norm = maxminnorm(var_rot.reshape(-1, 1))
    print('finish')
    return var_trans_norm, var_rot_norm


def find_var_peak(var_transdata, var_rotdata):
    trans_mean = var_transdata.mean()
    var_trans_peak_ind, _ = find_peaks(var_transdata.flatten(), prominence = [0.4,1],height=trans_mean + 0.2, distance = 30)
    var_rot_peak_ind, _ = find_peaks(var_rotdata.flatten(), height=0.5, prominence=[0.5, 1])

    var_trans_peak_indcwt = find_peaks_cwt(var_transdata.flatten(), widths=np.arange(10,50))
    var_rot_peak_indcwt = find_peaks_cwt(var_rotdata.flatten(), widths=np.arange(50,100))

    return var_trans_peak_ind, var_trans_peak_indcwt, var_rot_peak_ind, var_rot_peak_indcwt


def write_excel(sheetname, data, colp):
    sheetname.write_column(2, colp, data)

def segment_connect_acc(start_ind, end_ind, distance, axnum):
    for segments in range(len(end_ind)):
        start_point = start_ind[segments]
        end_point = end_ind[segments]
        axnum.plot(start_point+1, distance[start_point+1], "ro", ms=5,label= 'Ground Truth')
        axnum.plot(end_point+1, distance[end_point+1], "mo", ms=5)
        connectx = [start_point+1, end_point+1]
        connecty = [distance[start_point+1], distance[end_point+1]]
        axnum.plot(connectx, connecty, 'm--')

############################################
if __name__ == '__main__':
    file_org = "./result figures/original trans rot/"
    file_var = "./result figures/variation trans rot/"
    file_neg_org = "./result figures/negative original trans rot/"
    if not os.path.exists(file_org):
        # os.mkdir创建一个，os.makedirs可以创建路径上多个
        os.makedirs(file_org)
    if not os.path.exists(file_var):
        os.makedirs(file_var)
    if not os.path.exists(file_neg_org):
        os.makedirs(file_neg_org)


    tasks = ['Suturing', 'Needle Passing', 'Knot Tying']
    for task in tasks:
        rootPath = task + '/'
        filePath = rootPath + "kinematics/AllGestures/"
        all_filename = os.listdir(filePath)
        all_filename.sort()

        fileName = "./result file/potensial segment points.xls"  # 工作簿名字
        workbook = xw.Workbook(fileName)  # 创建工作簿

        for n in all_filename[::10]:
            print(n)
            start = []
            end = []
            seg_label = open(rootPath + "transcriptions/" + n, "r")
            f1 = open(filePath + n)
            print('working with file：', n)
            for lines in seg_label.readlines():
                # print(lines)
                column = lines.split(" ")
                column = [i for i in column if (len(str(i)) != 0)]
                start.append(column[0])
                end.append(column[1])

            start = np.array(start).astype(int)
            end = np.array(end).astype(int)
            end = np.delete(end, [-1])
            ''' ############### read trans and rot data for left and right hand ###'''
            translation_l, rotation_l, rot_vl, trans_vl, grip_vl = [], [], [], [], []
            translation_r, rotation_r, rot_vr, trans_vr, grip_vr = [], [], [], [], []

            for lines in f1.readlines():
                # print(lines)
                column = lines.split(" ")
                column = [i for i in column if (len(str(i)) != 0)]
                for line in column[0:3]:  ## translation
                    translation_l.append(line)
                for line in column[3:12]:  ## rotation matrix
                    rotation_l.append(line)
                for line in column[12:15]:
                    trans_vl.append(line)
                for line in column[15:18]:
                    rot_vl.append(line)
                for line in column[18]:
                    grip_vl.append(line)

                for line in column[19:22]:  ## translation for right hand
                    translation_r.append(line)
                for line in column[22:31]:  ## rotation matrix for right hand
                    rotation_r.append(line)
                for line in column[31:34]:
                    trans_vr.append(line)
                for line in column[34:37]:
                    rot_vr.append(line)
                for line in column[37]:
                    grip_vr.append(line)

            print(len(translation_l) / 3)



            translation_l,trans_vl= KALMAN_filter(translation_l,trans_vl,6)
            translation_r,trans_vr = KALMAN_filter(translation_r,trans_vr,6)
            print('Done kalman filter')

            eudist_norm_l = two_trans_eudist(translation_l)
            eudist_norm_r = two_trans_eudist(translation_r)
            eudist_norm = np.concatenate((eudist_norm_l, eudist_norm_r), axis=1)


            dist_rot_norm_l = two_rot_eudist(rotation_l)
            dist_rot_norm_r = two_rot_eudist(rotation_r)
            dist_rot_norm = np.concatenate((dist_rot_norm_l, dist_rot_norm_r), axis=1)

            dist_rot_norm_l = dist_rot_norm_l.flatten()
            dist_rot_norm_l = savgol_filter(dist_rot_norm_l,321, 1)
            dist_rot_norm_r = dist_rot_norm_r.flatten()
            dist_rot_norm_r = savgol_filter(dist_rot_norm_r, 321, 1)

            eudist_norm_l = eudist_norm_l.flatten()
            eudist_norm_l = savgol_filter(eudist_norm_l, 59, 2)
            eudist_norm_r = eudist_norm_r.flatten()
            eudist_norm_r = savgol_filter(eudist_norm_r, 59, 2)

            ''' #########################trans_index, trans_index_cwt, rot_index, rot_index_cwt ##################'''
            trans_peakl_ind, trans_peakl_indcwt, rot_peakl_ind, rot_peakl_indcwt = find_org_peaks(eudist_norm_l,
                                                                                                  dist_rot_norm_l)
            trans_peakr_ind, trans_peakr_indcwt, rot_peakr_ind, rot_peakr_indcwt = find_org_peaks(eudist_norm_r,
                                                                                                  dist_rot_norm_r)

            negeudist_norm_l = maxminnorm((-eudist_norm_l).reshape(-1, 1))
            negeudist_norm_r = maxminnorm(-eudist_norm_r.reshape(-1, 1))
            negrot_norm_l = maxminnorm((-dist_rot_norm_l).reshape(-1, 1))
            negrot_norm_r = maxminnorm((-dist_rot_norm_r).reshape(-1, 1))

            trans_peakl_ind1, trans_peakl_indcwt1, rot_peakl_ind1, rot_peakl_indcwt1 = find_neg_org_peaks(negeudist_norm_l,negrot_norm_l)
            trans_peakr_ind1, trans_peakr_indcwt1, rot_peakr_ind1, rot_peakr_indcwt1 = find_neg_org_peaks(negeudist_norm_r,negrot_norm_r)
            # negeudist_norm_r = eudist_norm_r.reshape(-1, 1)


            fig3 = plt.figure(figsize=(40, 15))
            ax3 = plt.subplot(221)
            plt.plot(eudist_norm_l[10:], color='#6495ED')
            ax3.plot(trans_peakl_indcwt, eudist_norm_l[trans_peakl_indcwt], "xr", ms=8)
            segment_connect(start, end, eudist_norm_l, ax3)
            plt.legend(['left translation distance cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax3 = plt.subplot(222)
            plt.plot(eudist_norm_r[10:], color='#6495ED')
            ax3.plot(trans_peakr_indcwt, eudist_norm_r[trans_peakr_indcwt], "xr", ms=8)
            segment_connect(start, end, eudist_norm_r, ax3)
            plt.legend(['right translation distance cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax3 = plt.subplot(223)
            plt.plot(dist_rot_norm_l[10:], color='#6495ED');
            ax3.plot(rot_peakl_indcwt, dist_rot_norm_l[rot_peakl_indcwt], "xr", ms=8)
            plt.legend(['left rotation distance cwt'],fontsize = 20)
            segment_connect(start, end, dist_rot_norm_l, ax3)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax3 = plt.subplot(224)
            plt.plot(dist_rot_norm_r[10:], color='#6495ED')
            ax3.plot(rot_peakr_indcwt, dist_rot_norm_r[rot_peakr_indcwt], "xr", ms=8)
            segment_connect(start, end, dist_rot_norm_r, ax3)
            plt.xlabel('frame')
            plt.ylabel('normalized distance')
            plt.legend(['right rotation distance cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            name = n[0:-4]
            plt.savefig(file_org + 'org trans rot cwt_' + n[0:-4])

            plt.clf()
            plt.close()

            '''##########################          variance-based              ##################'''
            ## define one new frame
            new_trans_l, new_rot_l = generate_new_frames(50, translation_l, rotation_l)
            new_trans_r, new_rot_r = generate_new_frames(50, translation_r, rotation_r)

            var_transl_norm, var_rotl_norm = calculate_var(new_trans_l, new_rot_l)
            var_transr_norm, var_rotr_norm = calculate_var(new_trans_r, new_rot_r)

            var_rotl_norm = var_rotl_norm.flatten()
            var_rotl_norm = savgol_filter(var_rotl_norm, 89, 3)
            var_rotr_norm = var_rotr_norm.flatten()
            var_rotr_norm = savgol_filter(var_rotr_norm, 89, 3)

            var_transl_norm = var_transl_norm.flatten()
            var_transl_norm = savgol_filter(var_transl_norm, 89, 3)
            var_transr_norm = var_transr_norm.flatten()
            var_transr_norm = savgol_filter(var_transr_norm, 89, 3)

            '''##########################          variance-based -  plot left hand           ##################'''
            var_trans_peakl_ind, var_trans_peakl_indcwt, var_rot_peakl_ind, var_rot_peakl_indcwt = find_var_peak(
                var_transl_norm, var_rotl_norm)
            var_trans_peakr_ind, var_trans_peakr_indcwt, var_rot_peakr_ind, var_rot_peakr_indcwt = find_var_peak(
                var_transr_norm, var_rotr_norm)

            ############################################################################################33

            nt = -var_transl_norm
            nr = -var_rotl_norm
            var_transl_negnorm = maxminnorm(nt.reshape(-1, 1))
            var_rotl_negnorm = maxminnorm(nr.reshape(-1, 1))
            var_transr_negnorm = maxminnorm((-var_transr_norm).reshape(-1, 1))
            var_rotr_negnorm = maxminnorm((-var_rotr_norm).reshape(-1, 1))

            var_trans_peakl_negind, var_trans_peakl_negindcwt, var_rot_peakl_negind, var_rot_peakl_negindcwt = find_var_peak(
                var_transl_negnorm, var_rotl_negnorm)
            var_trans_peakr_negind, var_trans_peakr_negindcwt, var_rot_peakr_negind, var_rot_peakr_negindcwt = find_var_peak(
                var_transr_negnorm, var_rotr_negnorm)

    ##########################

            fig6 = plt.figure(figsize=(40, 15))
            ax6 = plt.subplot(221)
            ax6.plot(var_transl_norm)
            ax6.plot(var_trans_peakl_indcwt, var_transl_norm[var_trans_peakl_indcwt], 'xr');
            segment_connect(start, end, var_transl_norm, ax6);
            plt.legend(['rotation var left cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax6 = plt.subplot(222)
            ax6.plot(var_transr_norm)
            ax6.plot(var_trans_peakr_indcwt, var_transr_norm[var_trans_peakr_indcwt], 'xr');
            segment_connect(start, end, var_transr_norm, ax6);
            plt.legend(['translation var right cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax6 = plt.subplot(223)
            ax6.plot(var_transl_norm)
            ax6.plot(var_trans_peakl_ind, var_transl_norm[var_trans_peakl_ind], 'xr');
            segment_connect(start, end, var_transl_norm, ax6);
            plt.legend(['translation var left'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax6 = plt.subplot(224)
            ax6.plot(var_transr_norm)
            ax6.plot(var_trans_peakr_ind, var_transr_norm[var_trans_peakr_ind], 'xr');
            segment_connect(start, end, var_transr_norm, ax6);
            plt.legend(['translation var right'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)


            plt.savefig(file_var + 'trans var' + n[0:-4])

            plt.clf()
            plt.close()
            ####################################333
            fig = plt.figure(figsize=(40, 15))
            ax5 = plt.subplot(221)
            ax5.plot(var_rotl_norm)
            ax5.plot(var_rot_peakl_indcwt, var_rotl_norm[var_rot_peakl_indcwt], 'xr');
            segment_connect(start, end, var_rotl_norm, ax5);
            plt.legend(['rotation var left cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax5 = plt.subplot(222)
            ax5.plot(var_rotr_norm)
            ax5.plot(var_rot_peakr_indcwt, var_rotr_norm[var_rot_peakr_indcwt], 'xr');
            segment_connect(start, end, var_rotr_norm, ax5);
            plt.legend(['rotation var right cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax5 = plt.subplot(223)
            ax5.plot(var_rotl_negnorm)
            ax5.plot(var_rot_peakl_negindcwt, var_rotl_negnorm[var_rot_peakl_negindcwt], 'xr');
            segment_connect(start, end, var_rotl_negnorm, ax5);
            plt.legend(['negative rotation var left cwt'],fontsize = 20)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)

            ax5 = plt.subplot(224)
            ax5.plot(var_rotr_negnorm)
            ax5.plot(var_rot_peakr_negindcwt, var_rotr_negnorm[var_rot_peakr_negindcwt], 'xr')
            segment_connect(start, end, var_rotr_negnorm, ax5)
            plt.xlabel('frame',fontsize = 20)
            plt.ylabel('normalized distance',fontsize = 20)
            plt.legend(['negative rotation var right cwt'],fontsize = 20)

            plt.savefig(file_var + 'negvar cwt' + n[0:-4])

            plt.clf()
            plt.close()

            """#############################    write in file  ##############"""

            worksheet1 = workbook.add_worksheet(n[0:-4])  # 创建子表
            worksheet1.activate()  # 激活表

            head_data = np.array(
                ['trans_peakl_ind', 'trans_peakr_ind', 'rot_peakl_ind', 'rot_peakr_ind',
                'trans_peakl_indcwt', 'trans_peakr_indcwt', 'rot_peakl_indcwt', 'rot_peakr_indcwt', #8
                'start',
                'var_rot_peakl_indcwt', 'var_rot_peakl_negindcwt', 'var_rot_peakr_indcwt', 'var_rot_peakr_negindcwt',
                'var_trans_peakl_indcwt', 'var_trans_peakl_negindcwt', 'var_rot_peakl_ind', 'var_rot_peakl_negind',
                'var_trans_peakr_indcwt', 'var_trans_peakr_negindcwt', 'var_rot_peakr_ind', 'var_rot_peakr_negind'])

            # all_data = [] 　
            all_data = [trans_peakl_ind, trans_peakr_ind, rot_peakl_ind, rot_peakr_ind,
                        trans_peakl_indcwt, trans_peakr_indcwt, rot_peakl_indcwt, rot_peakr_indcwt, #8
                        start,
                        var_rot_peakl_indcwt, var_rot_peakl_negindcwt, var_rot_peakr_indcwt, var_rot_peakr_negindcwt,
                        rot_peakl_indcwt1, rot_peakr_indcwt1, #14,15
                        var_trans_peakl_indcwt, var_trans_peakl_negindcwt, var_rot_peakl_ind, var_rot_peakl_negind,
                        var_trans_peakr_indcwt, var_trans_peakr_negindcwt, var_rot_peakr_ind, var_rot_peakr_negind ]


            transl_ind = np.concatenate((trans_peakl_ind, trans_peakl_indcwt))
            transr_ind = np.concatenate((trans_peakr_ind, trans_peakr_indcwt))

            column_num = 0
            for d in all_data:
                write_excel(worksheet1, d, column_num)
                column_num += 1

            worksheet1.write_row("A1", head_data)

        workbook.close()

        # plt.show()
