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

'''###################### translation velocity Euclidean Distance ####################'''


def safe_concatenate(X, W):
    """
    Checks if X is None before concatenating W
    to X along specified axis (0 by default)
	"""
    if X is None:
        return W
    else:
        return np.vstack((X, W))


def KALMAN_filter(measurement1, measurement2, n):
    cor = None
    v = None
    measurement1 = np.array(measurement1, np.float32).reshape(-1, 3)
    measurement2 = np.array(measurement2, np.float32).reshape(-1, 3)
    kalman = cv2.KalmanFilter(6, n)
    kalman.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                        [0, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 0, 0, 1],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], np.float32)  # 转移矩阵 A
    kalman.measurementMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                         [0, 1, 0, 0, 1, 0],
                                         [0, 0, 1, 0, 0, 1],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)  # 测量矩阵    H
    kalman.measurementNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                           [0, 1, 0, 0, 0, 0],
                                           [0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]], np.float32) * 100  # 测量噪声        R
    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32) * 1e-5  # 过程噪声 Q
    measurement = np.hstack((measurement1, measurement2))
    for i in range(len(measurement)):
        mes = np.reshape(measurement[i, :], (n, 1))
        # measurement = np.array([[np.float32(x)], [np.float32(y)]])
        c = kalman.correct(mes)
        # cor.append(c)
        current_prediction = kalman.predict()
        cor = safe_concatenate(cor, current_prediction.flatten()[0:3])
        v = safe_concatenate(v, current_prediction.flatten()[3:6])
    return cor, v


def max_min_norm(array):
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


def trans_eudist(translation):
    translation = np.array(translation).astype(float).reshape(-1, 3) * 1000
    eudist = [0 for _ in range(int(len(translation) - 1))]
    for index in range(0, len(eudist)):
        eudist_res = eucldist_vectorized(translation[index], translation[index + 1])
        eudist[index] = eudist_res
    eudist = np.array(eudist).reshape(-1, 1)
    eudistance_norm = max_min_norm(eudist).reshape(-1, 1)
    return eudistance_norm


def quaternion(rotation_matrix):
    rotation_matrix = np.array(rotation_matrix)
    arr_euler = rotation_matrix.reshape(3, -1)
    r3 = R.from_matrix(arr_euler)
    qua = r3.as_quat()
    ## calculate rotation distance dist_rot ####
    # d.write(str(qua[0]) + ' ' + str(qua[1]) + ' ' + str(qua[2]) + ' ' + str(qua[3]) + '\n')
    return qua


def rot_eudist(rotation):
    counter = 0
    qua = []
    # dist_rot = [0 for _ in range((int(len(rotation) / 9)))]
    dist_rot = []
    eps = 1e-6
    # rotation distance
    for index in range(len(rotation)):
        while counter < len(rotation) - 9:
            qua.append(quaternion(rotation[counter: counter + 9]))
            if len(qua) > 1:
                # a [0]= np.arccos(2 * np.inner(qua[i], qua[i - 1]) ** 2 - 1)
                dist_rot.append((np.arccos(2 * np.inner(qua[-1], qua[-2]) ** 2 - 1)))
                # dist_rot[counter - 1] = np.arccos(2 * np.inner(qua[counter], qua[counter - 1]) ** 2 - 1)
            counter = counter + 9
    dist_rot = np.array(dist_rot).reshape(-1, 1)
    dist_rot = np.nan_to_num(dist_rot)
    dist_rot_norm = max_min_norm(dist_rot).reshape(-1, 1)
    return dist_rot_norm


############################################### find prominence and corner ####################
def find_org_peaks(org_trans, org_rot):
    # org_trans = maxminnorm((-org_trans).reshape(-1, 1))
    # org_rot = maxminnorm((-org_rot).reshape(-1, 1))
    trans_index, _ = find_peaks(org_trans.flatten(), prominence=[0.05, 0.5], height=0.2)  # height = 0.05
    trans_index_cwt = find_peaks_cwt(org_trans.flatten(), widths=np.arange(1, 300))  # 原来（50，300）

    rot_index, _ = find_peaks(org_rot.flatten(), distance=60, prominence=0.1)
    rot_index_cwt = find_peaks_cwt(org_rot.flatten(), widths=np.arange(1, 300))  # 原来（50，300）

    return trans_index, trans_index_cwt, rot_index, rot_index_cwt


def find_neg_org_peaks(org_trans, org_rot):
    # org_trans = maxminnorm((-org_trans).reshape(-1, 1))
    # org_rot = maxminnorm((-org_rot).reshape(-1, 1))
    trans_index, _ = find_peaks(org_trans.flatten(), prominence=[0.05, 0.5], height=0.8)  # height = 0.05
    trans_index_cwt = find_peaks_cwt(org_trans.flatten(), widths=np.arange(1, 300))  # 原来（50，300）

    rot_index, _ = find_peaks(org_rot.flatten(), distance=60, prominence=0.1)
    rot_index_cwt = find_peaks_cwt(org_rot.flatten(), widths=np.arange(1, 300))  # 原来（50，300）

    return trans_index, trans_index_cwt, rot_index, rot_index_cwt


def segment_connect(start_ind, end_ind, distance, axnum):
    for segments in range(len(end_ind)):
        start_point = start_ind[segments]
        end_point = end_ind[segments]
        gt, = axnum.plot(start_point + 1, distance[start_point + 1], "ro", ms=5)
        axnum.plot(end_point + 1, distance[end_point + 1], "mo", ms=5)
        connectx = [start_point + 1, end_point + 1]
        connecty = [distance[start_point + 1], distance[end_point + 1]]
        seg_connect, = axnum.plot(connectx, connecty, 'm--')
    return seg_connect, gt,


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
    var_trans_norm = max_min_norm(var_trans.reshape(-1, 1))
    var_rotx = np.array(rot_x.var(axis=0))
    var_roty = np.array(rot_y.var(axis=0))
    var_rotz = np.array(rot_z.var(axis=0))
    var_rot = var_rotx + var_roty + var_rotz
    var_rot_norm = max_min_norm(var_rot.reshape(-1, 1))
    return var_trans_norm, var_rot_norm


def find_var_peak(var_transdata, var_rotdata):
    trans_mean = var_transdata.mean()
    var_trans_peak_ind, _ = find_peaks(var_transdata.flatten(), prominence=[0.4, 1], height=trans_mean + 0.2,
                                       distance=30)
    var_rot_peak_ind, _ = find_peaks(var_rotdata.flatten(), height=0.5, prominence=[0.5, 1])

    var_trans_peak_indcwt = find_peaks_cwt(var_transdata.flatten(), widths=np.arange(10, 50))
    var_rot_peak_indcwt = find_peaks_cwt(var_rotdata.flatten(), widths=np.arange(50, 100))

    return var_trans_peak_ind, var_trans_peak_indcwt, var_rot_peak_ind, var_rot_peak_indcwt


def write_excel(sheetname, data, colp):
    sheetname.write_column(2, colp, data)


def segment_connect_acc(start_ind, end_ind, distance, axnum):
    for segments in range(len(end_ind)):
        start_point = start_ind[segments]
        end_point = end_ind[segments]
        axnum.plot(start_point + 1, distance[start_point + 1], "ro", ms=5, label='Ground Truth')
        axnum.plot(end_point + 1, distance[end_point + 1], "mo", ms=5)
        connectx = [start_point + 1, end_point + 1]
        connecty = [distance[start_point + 1], distance[end_point + 1]]
        axnum.plot(connectx, connecty, 'm--')


def get_seg_label(kinematic_filename):
    start = []
    end = []
    seg_label = open(rootPath + "transcriptions/" + kinematic_filename, "r")
    for lines in seg_label.readlines():
        # print(lines)
        column = lines.split(" ")
        column = [i for i in column if (len(str(i)) != 0)]
        start.append(column[0])
        end.append(column[1])
    start = np.array(start).astype(int)
    end = np.array(end).astype(int)
    end = np.delete(end, [-1])
    return start, end


def get_kinematic_data():
    translation_left, rotation_left, rot_vleft, trans_vleft, grip_vleft = [], [], [], [], []
    translation_right, rotation_right, rot_vright, trans_vright, grip_vright = [], [], [], [], []
    kinematics_filepath = open(filePath + kinematics_file)
    for lines in kinematics_filepath.readlines():
        # print(lines)
        column = lines.split(" ")
        column = [i for i in column if (len(str(i)) != 0)]
        for line in column[0:3]:  ## translation
            translation_left.append(line)
        for line in column[3:12]:  ## rotation matrix
            rotation_left.append(line)
        for line in column[12:15]:
            trans_vleft.append(line)
        for line in column[15:18]:
            rot_vleft.append(line)
        for line in column[18]:
            grip_vleft.append(line)

        for line in column[19:22]:  ## translation for right hand
            translation_right.append(line)
        for line in column[22:31]:  ## rotation matrix for right hand
            rotation_right.append(line)
        for line in column[31:34]:
            trans_vright.append(line)
        for line in column[34:37]:
            rot_vright.append(line)
        for line in column[37]:
            grip_vright.append(line)
    print(len(translation_left) / 3)
    return translation_left, rotation_left, trans_vleft, \
        translation_right, rotation_right, trans_vright


def data_filters(trans_data, rot_data, trans_vdata):
    trans_data, trans_v_data = KALMAN_filter(trans_data, trans_vdata, 6)
    eudist_norm, dist_rot_norm \
        = trans_rot_eudistance(trans_data, rot_data)
    # dist_rot_norm = np.concatenate((dist_rot_norm_l, dist_rot_norm_r), axis=1)
    dist_rot_filter = sav_filter(dist_rot_norm, 321, 1)
    eudist_filter = sav_filter(eudist_norm, 59, 2)
    return eudist_filter, dist_rot_filter


def sav_filter(data, window_length, polyorder):
    data = data.flatten()
    output = savgol_filter(data, window_length, polyorder)
    return output


def trans_rot_eudistance(trans, rot):
    eudistance_norm = trans_eudist(trans)
    distance_rot_norm = rot_eudist(rot)
    return eudistance_norm, distance_rot_norm


def plot_figs(indxdata_dict, data_dict, fig_name, fig_path):
    fig = plt.figure(figsize=(40, 15), dpi=600)
    i = 0
    for dict1, dict2 in zip(indxdata_dict.items(), data_dict.items()):
        ax = fig.add_subplot(2, 2, i + 1)
        k_line, = plt.plot(dict2[1][5:], color='#6495ED')
        estimate, = ax.plot(dict1[1], dict2[1][dict1[1]], "xr", ms=8)
        connect_line, ground_truth = segment_connect(start_timestamp, end_timestamp, dict2[1], ax)
        plt.legend([k_line, estimate, ground_truth], [dict1[0], "estimated points", "ground truth points"], fontsize=15)
        ax.tick_params(axis='both', labelsize=20)
        plt.xlabel('frame', fontsize=25)
        plt.ylabel('normalized distance', fontsize=25)
        i += 1
    plt.savefig(fig_path + fig_name[0:-4])
    plt.clf()
    plt.close()


############################################
if __name__ == '__main__':
    homepath = os.getcwd() + "/"
    img_org_path = "./result figures/original trans rot/"
    img_var_path = "./result figures/variation trans rot/"
    img_neg_org_path = "./result figures/negative original trans rot/"
    res_file = "./result file/"
    dst_files = [img_org_path, img_var_path, img_neg_org_path, res_file]
    for dst in dst_files:
        if not os.path.exists(dst):
            # build files/dir if not exist
            os.makedirs(dst)

    fileName = "./result file/potensial segment points.xlsx"  # 工作簿名字
    workbook = xw.Workbook(fileName)

    tasks = ['Suturing', 'Needle_Passing', 'Knot_Tying']
    for task in tasks:
        rootPath = homepath + task + '/'
        filePath = rootPath + "kinematics/AllGestures/"
        gesture_filename = os.listdir(filePath)
        gesture_filename.sort()
        # 创建工作簿

        for kinematics_file in gesture_filename[::10]:
            print(kinematics_file)
            start_timestamp, end_timestamp = get_seg_label(kinematics_file)

            ''' ############### read trans and rot data for left and right hand #######'''
            translation_l, rotation_l, trans_vl, translation_r, rotation_r, trans_vr = get_kinematic_data()

            eudist_norm_l, dist_rot_norm_l = data_filters(translation_l, rotation_l, trans_vl)
            dist_rot_norm_r, eudist_norm_r = data_filters(translation_r, rotation_r, trans_vr)

            ''' ################ trans_index, trans_index_cwt, rot_index, rot_index_cwt ##################'''
            trans_peakl_ind, trans_peakl_indcwt, rot_peakl_ind, rot_peakl_indcwt = find_org_peaks(eudist_norm_l,
                                                                                                  dist_rot_norm_l)
            trans_peakr_ind, trans_peakr_indcwt, rot_peakr_ind, rot_peakr_indcwt = find_org_peaks(eudist_norm_r,
                                                                                                  dist_rot_norm_r)

            negeudist_norm_l = max_min_norm((-eudist_norm_l).reshape(-1, 1))
            negeudist_norm_r = max_min_norm(-eudist_norm_r.reshape(-1, 1))
            negrot_norm_l = max_min_norm((-dist_rot_norm_l).reshape(-1, 1))
            negrot_norm_r = max_min_norm((-dist_rot_norm_r).reshape(-1, 1))

            trans_peakl_ind1, trans_peakl_indcwt1, rot_peakl_ind1, rot_peakl_indcwt1 \
                = find_neg_org_peaks(negeudist_norm_l, negrot_norm_l)
            trans_peakr_ind1, trans_peakr_indcwt1, rot_peakr_ind1, rot_peakr_indcwt1 \
                = find_neg_org_peaks(negeudist_norm_r, negrot_norm_r)

            indxcwt_dict = {"left translation distance cwt": trans_peakl_indcwt,
                            "right translation distance cwt": trans_peakr_indcwt,
                            "left rotation distance cwt": rot_peakl_indcwt,
                            "right rotation distance cwt": rot_peakr_indcwt}
            indxorg_dict = {"trans_peakl": trans_peakl_ind, "trans_peakr": trans_peakr_ind,
                            "rot_peakl": rot_peakl_ind, "rot_peakr": rot_peakr_ind}
            data_dict = {"left translation data": eudist_norm_l,
                         "right translation data": eudist_norm_r,
                         "left rotation data": dist_rot_norm_l,
                         "right rotation data": dist_rot_norm_r
                         }
            fig_name = kinematics_file[0:-4] + 'org trans rot cwt_'
            plot_figs(indxcwt_dict, data_dict, fig_name, img_org_path)

            '''##########################          variance-based              ##################'''
            ## define one new frame
            new_trans_l, new_rot_l = generate_new_frames(50, translation_l, rotation_l)
            new_trans_r, new_rot_r = generate_new_frames(50, translation_r, rotation_r)

            var_transl, var_rotl = calculate_var(new_trans_l, new_rot_l)
            var_transr, var_rotr = calculate_var(new_trans_r, new_rot_r)

            var_rotl_norm = sav_filter(var_rotl, 89, 3)
            var_rotr_norm = sav_filter(var_rotr, 89, 3)
            var_transl_norm = sav_filter(var_transl, 89, 3)
            var_transr_norm = sav_filter(var_transr, 89, 3)

            '''##########################          variance-based -  plot left hand           ##################'''
            var_trans_peakl_ind, var_trans_peakl_indcwt, var_rot_peakl_ind, var_rot_peakl_indcwt = find_var_peak(
                var_transl_norm, var_rotl_norm)
            var_trans_peakr_ind, var_trans_peakr_indcwt, var_rot_peakr_ind, var_rot_peakr_indcwt = find_var_peak(
                var_transr_norm, var_rotr_norm)

            nt = -var_transl_norm
            nr = -var_rotl_norm
            var_transl_negnorm = max_min_norm(nt.reshape(-1, 1))
            var_rotl_negnorm = max_min_norm(nr.reshape(-1, 1))
            var_transr_negnorm = max_min_norm((-var_transr_norm).reshape(-1, 1))
            var_rotr_negnorm = max_min_norm((-var_rotr_norm).reshape(-1, 1))

            var_trans_peakl_negind, var_trans_peakl_negindcwt, var_rot_peakl_negind, var_rot_peakl_negindcwt \
                = find_var_peak(var_transl_negnorm, var_rotl_negnorm)
            var_trans_peakr_negind, var_trans_peakr_negindcwt, var_rot_peakr_negind, var_rot_peakr_negindcwt \
                = find_var_peak(var_transr_negnorm, var_rotr_negnorm)

            var_indx_dict = {"left translation variance cwt": var_trans_peakl_indcwt,
                             "right translation variance cwt": var_trans_peakr_indcwt,
                             "negative left variance var cwt": var_trans_peakl_negindcwt,
                             "negative right variance var cwt": var_trans_peakr_negindcwt}

            var_data_dict = {"left variance data": var_transl_norm,
                             "right variance data": var_transr_norm,
                             "left variance data": var_transl_negnorm,
                             "right variance data": var_transr_negnorm
                             }
            fig_name = kinematics_file[0:-4] + 'trans var cwt'
            plot_figs(indxcwt_dict, data_dict, fig_name, img_var_path)

            """#############################    write in file  ######################"""

            worksheet1 = workbook.add_worksheet(kinematics_file[0:-4])  # 创建子表
            worksheet1.activate()  # 激活表

            head_data = np.array(
                ['trans_peakl_ind', 'trans_peakr_ind', 'rot_peakl_ind', 'rot_peakr_ind',
                 'trans_peakl_indcwt', 'trans_peakr_indcwt', 'rot_peakl_indcwt', 'rot_peakr_indcwt',  # 8
                 'start',
                 'var_rot_peakl_indcwt', 'var_rot_peakl_negindcwt', 'var_rot_peakr_indcwt', 'var_rot_peakr_negindcwt',
                 'var_trans_peakl_indcwt', 'var_trans_peakl_negindcwt', 'var_rot_peakl_ind', 'var_rot_peakl_negind',
                 'var_trans_peakr_indcwt', 'var_trans_peakr_negindcwt', 'var_rot_peakr_ind', 'var_rot_peakr_negind'])

            all_data = [trans_peakl_ind, trans_peakr_ind, rot_peakl_ind, rot_peakr_ind,
                        trans_peakl_indcwt, trans_peakr_indcwt, rot_peakl_indcwt, rot_peakr_indcwt,  # 8
                        start_timestamp,
                        var_rot_peakl_indcwt, var_rot_peakl_negindcwt, var_rot_peakr_indcwt, var_rot_peakr_negindcwt,
                        rot_peakl_indcwt1, rot_peakr_indcwt1,  # 14,15
                        var_trans_peakl_indcwt, var_trans_peakl_negindcwt, var_rot_peakl_ind, var_rot_peakl_negind,
                        var_trans_peakr_indcwt, var_trans_peakr_negindcwt, var_rot_peakr_ind, var_rot_peakr_negind]

            transl_ind = np.concatenate((trans_peakl_ind, trans_peakl_indcwt))
            transr_ind = np.concatenate((trans_peakr_ind, trans_peakr_indcwt))

            column_num = 0
            for d in all_data:
                write_excel(worksheet1, d, column_num)
                column_num += 1
            worksheet1.write_row("A1", head_data)

    workbook.close()
