import os
from sklearn.metrics import  recall_score, precision_score, confusion_matrix,accuracy_score
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np

colors = {0: '#FFB6C1',1:'#CC99FF',
             2:'#99CCFF', 3:'#66CCFF',4:'#FFCC00',5:'#FF9966', 6: '#FFFF00',7: '#FF6666', 8: '#99FF99',
            9:'#99FFFF',10:'#99FF00',11:'#CCFFFF',12:'#99CCCC', 13:'#CC66CC',14:'#CCCC99',15:'#99CC99',16:'#FF9966',
             17:'#CC9966',18:'#999966',19:'#FF66FF',
             20: '#FFFFCC'}

res_path = 'D:/J/Transformer_mix/score/'
all_label_path = 'D:/J/Transformer_mix/result file/'
all_labels_file = os.listdir(all_label_path)
all_precision = dict()
all_recall = dict()
all_true = []
all_predict = []
for label in all_labels_file:
    label_file = open(all_label_path + label)
    labels_true = []
    labels_pred = []
    frames = []
    for lines in label_file.readlines():
        column = lines.split(" ")
        if int(column[0])!=0:
            labels_true.append(int(column[0]))
            labels_pred.append(int(column[1]))
            frames.append(int(column[2]))
        else: continue

    all_true.extend(labels_true)
    all_predict.extend(labels_pred)

    log = open(res_path + label[:-4] + '.txt', "w")
    label_based_scores = {}
    for ave in ["micro", "macro", "weighted"]:
        key = "precision_" + ave
        score_1 = precision_score(labels_true, labels_pred, average=ave, zero_division=0)
        if key in all_precision:
            all_precision[key] += score_1
        else:
            all_precision[key] = score_1
        label_based_scores[key] = score_1
        log.write("%3.3f        %s\n" % (score_1, key))

        key = "recall_" + ave
        score_1 = recall_score(labels_true, labels_pred, average=ave, zero_division=0)
        label_based_scores[key] = score_1
        if key in all_recall:
            all_recall[key] += score_1
        else:
            all_recall[key] = score_1
        log.write("%3.3f        %s\n" % (score_1, key))

    """############################ plot for result #################################################"""
    pre_start = []
    pre_end = []
    pre_start.append(frames[0])
    plt_label = []
    plt_label.append(labels_pred[0])
    list_pre_color = []
    list_pre_color.append(colors[labels_pred[0]])
    for pre_idx in range(len(labels_pred) - 1):
        if labels_pred[pre_idx + 1] != labels_pred[pre_idx]:
            plt_label.append(labels_pred[pre_idx + 1])
            pre_start.append(frames[pre_idx])
            pre_end.append(frames[pre_idx] - 1)
            list_pre_color.append(colors[labels_pred[pre_idx + 1]])

    pre_end.append(frames[-1])
    list_start_end = []
    for ele in range(len(pre_end)):
        list_start_end.append((pre_start[ele], pre_end[ele]))

    gt_start = []
    gt_end = []
    gt_start.append(frames[0])
    plt_gtlabel = []
    plt_gtlabel.append(labels_true[0])
    list_gt_color = []
    list_gt_color.append(colors[labels_true[0]])
    for gt_idx in range(len(labels_true) - 1):
        if labels_true[gt_idx + 1] != labels_true[gt_idx]:
            plt_gtlabel.append(labels_true[gt_idx + 1])
            gt_start.append(frames[gt_idx])
            gt_end.append(frames[gt_idx] - 1)
            list_gt_color.append(colors[labels_true[gt_idx + 1]])

    gt_end.append(frames[-1])
    list_gt_start_end = []
    for ele in range(len(gt_end)):
        list_gt_start_end.append((gt_start[ele], gt_end[ele]))

    # fig, ax = plt.subplots(figsize=(30, 5))
    # ax.broken_barh(list_start_end, [10, 0.5], facecolor=tuple(list_pre_color))
    # for idx, t in enumerate(pre_start):
    #     plt.text(t + 10, 10.5, 'G%d' % plt_label[idx], color='k', fontsize=12)
    #
    # ax.broken_barh(list_gt_start_end, [11, 0.5], facecolor=tuple(list_gt_color))
    # for idx, t in enumerate(gt_start):
    #     plt.text(t + 10, 11.5, 'G%d' % plt_gtlabel[idx], color='k', fontsize=12)
    #
    # plt.ylim(9.5, 12)
    # plt.xlim(0, pre_end[-1])
    # plt.xlabel('frame',fontsize = 20)
    #
    # ax.set_title('Final result for estimated segment points + Transformer based model : '+ label[15:-4],fontsize = 18)
    # ax.set_yticks([10.5, 11.5])
    # ax.set_yticklabels(['Final result', 'Ground truth'],fontsize = 20)
    # plt.savefig('D:/J/Transformer_mix/classification figure/' + label[:-4] + '.png')
    # plt.cla()
    # plt.close("all")

# print(all_precision)
# print(all_recall)
for ave in ["micro", "macro", "weighted"]:
    keyp = "precision_" + ave
    keyr = "recall_" + ave
    print('mean of ' +keyp  +'is: ', all_precision[keyp]/len(all_labels_file))
    print('mean of ' +keyr  +' is: ', all_recall[keyr]/len(all_labels_file))

acc_score = accuracy_score(all_true, all_predict)
print('accuracy score is: ', acc_score)


#labels表示你不同类别的代号，比如这里的demo中有13个类别
labels = ['1', '2', '3', '4', '5', '6', '8','9','10', '11','12','13','14','15']
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap='Blues'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if set(all_true) == set(all_predict):
    print(set(all_true))

cm = confusion_matrix(all_true, all_predict)
np.set_printoptions(precision=4)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm_normalized)
plt.figure(figsize=(15, 15), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.4f" % (c,), color='red', fontsize=15, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('./confusion_matrix.png', format='png')
plt.close()
plt.clf()

