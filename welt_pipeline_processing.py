import numpy as np
import pandas as pd
import os
import lap
import sqlite3
import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import scipy.stats as st

meas_path_in = './dataset/01_TrackTraining.csv'
meas_path_out = './dataset/01_TrackTraining_mod_world.csv'
meas_path_out_x10 = './dataset/01_TrackTraining_mod_world_x10.csv'
meas_path_out_x100 = './dataset/01_TrackTraining_mod_world_x100.csv'
out_path_x10_2 = './dataset/01_TrackTraining_mod_world_x10_all_tracks.csv'

out_path_x10_f30 = './dataset/01_TrackTraining_mod_world_x10_more_tracks_f30.csv'
out_path_x10_f100 = './dataset/01_TrackTraining_mod_world_x10_more_tracks_f100.csv'
out_path_x10_f728 = './dataset/01_TrackTraining_mod_world_x10_more_tracks_f728.csv'
out_path_x10_f728_less = './dataset/01_TrackTraining_mod_world_x10_more_tracks_f728_f944_less.csv'
out_path_max_shelter = './dataset/01_TrackTraining_max_shelter.csv'


def process_detections_result(in_path, out_path):

    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')

    arr = np.asarray(csv)
    arr = arr[arr[:,1]<54]  # 2，4，8，11，12，14，16
    arr = arr[arr[:,0] < 17]
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    f = open(out_path, 'w')
    for row in res:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3], row[4]), file=f)  # changed sequence fid,tid,bbox,..


def process_detections_result_x100(in_path, out_path): # all pos_x,y * 100
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')

    arr = np.asarray(csv)
    arr = arr[arr[:, 1] < 54]  # 2，4，8，11，12，14，16
    arr = arr[arr[:, 0] < 17]
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    f = open(out_path, 'w')
    for row in res:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3]*100, row[4]*100), file=f)  # changed sequence fid,tid,bbox,..


def process_detections_result_x10(in_path, out_path): # all pos_x,y * 100
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')

    arr = np.asarray(csv)
    arr = arr[arr[:, 1] < 54]
    arr = arr[arr[:, 0] < 17]  # 2，4，8，11，12，14，16
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    f = open(out_path, 'w')
    for row in res:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3]*10, row[4]*10), file=f)


# select frame0 to frame 25 all tracks with world coordinate center and data all times 10
def process_detections_result_x10_all_tracks(in_path, out_path):  # all pos_x,y * 100
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')

    arr = np.asarray(csv)
    arr = arr[arr[:, 1] < 25]  # Frame
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    f = open(out_path, 'w')
    for row in res:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3]*10, row[4]*10), file=f)

# select frame0 to frame 30 covered tracks with world coordinate center and data all times 10
def process_detections_result_x10_more_tracks(in_path, out_path, id_sel_list):  # all pos_x,y * 10
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    l = []
    arr = np.asarray(csv)
    arr = arr[arr[:, 1] < 30]
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    for r in res:
        if r[1] in id_sel_list:
            l.append(r)

    l_arr = np.asarray(l)
    f = open(out_path, 'w')
    for row in l_arr:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3]*10, row[4]*10), file=f)


# situation1, l2+straight lane. select frame0 to frame 100 covered tracks with world coordinate center and data all times 10
def process_detections_result_x10_more_tracks_f100(in_path, out_path, id_sel_list):  # all pos_x,y * 10
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    l = []
    arr = np.asarray(csv)
    arr = arr[arr[:, 1] < 100]
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    for r in res:
        if r[1] in id_sel_list:
            l.append(r)
    l_arr = np.asarray(l)
    f = open(out_path, 'w')
    for row in l_arr:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3]*10, row[4]*10), file=f)


# situation2, l3,l4+straight lane. select frame0 to frame 100 covered tracks with world coordinate center
def process_detections_result_x10_more_tracks_f728_f950(in_path, out_path, id_sel_list):  # all pos_x,y * 10
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    l = []
    arr = np.asarray(csv)
    arr = arr[arr[:, 1] > 727]
    arr = arr[arr[:, 1] < 950]
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    for r in res:
        if r[1] in id_sel_list:
            l.append(r)
    l_arr = np.asarray(l)
    f = open(out_path, 'w')
    for row in l_arr:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3] * 10, row[4] * 10), file=f)


def get_max_shelter_length(in_path, out_path): # 添加统计功能
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    arr = np.asarray(csv)
    ids_all = arr[:,0]
    ids = np.unique(ids_all)
    all_length = []
    arr_pick_info_largest= []
    arr_pick_greater_than80 = []
    arr_40_80 = []
    arr_less_than_40 = []
    for id in ids:
        arr_pick = arr[arr[:, 0] == id]
        for i in range(len(arr_pick)-1):
            if arr_pick[i+1][1]-arr_pick[i][1]>2:
                all_length.append(arr_pick[i+1][1]-arr_pick[i][1])
            if arr_pick[i+1][1]-arr_pick[i][1]>50:
                arr_pick[i+1][2]= arr_pick[i+1][1]-arr_pick[i][1]
                arr_pick_info_largest.append(arr_pick[i+1])
            if arr_pick[i + 1][1] - arr_pick[i][1] < 40 and arr_pick[i + 1][1] - arr_pick[i][1] > 2:
                arr_less_than_40.append(arr_pick[i+1])
            if arr_pick[i + 1][1] - arr_pick[i][1] > 40 and arr_pick[i + 1][1] - arr_pick[i][1] < 80:
                arr_40_80.append(arr_pick[i + 1])
            if arr_pick[i + 1][1] - arr_pick[i][1] > 80:
                arr_pick_greater_than80.append(arr_pick[i+1])
    f = open(out_path, 'w')
    for row in arr_pick_info_largest:
        print('%d,%d,%d' % (
            row[0], row[1], row[2]), file=f)
    all_length.sort()
    return len(all_length), len(ids),len(arr_pick_greater_than80), len(arr_40_80), len(arr_less_than_40)

def create_test_file(in_path):
    col_use = ['ID', 'Time', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    array = np.asarray(csv)

    array = array[array[:, 1] < 50]
    array = array[array[:,0]<30]
    ids_new = np.unique(array[:, 0])
    out_path_ori = './dataset/01_TrackTraining_first50frames_for_data_cluster_original.csv'
    out_path_x10 = './dataset/01_TrackTraining_first50frames_for_data_cluster_x10.csv'
    f = open(out_path_ori, 'w')
    fx10 = open(out_path_x10, 'w')
    for row in array:
        print('%d,%.3f,%.3f' % (
            row[0], row[2], row[3]), file=f)

    for row in array:
        print('%d,%.3f,%.3f' % (
            row[0], row[2]*10, row[3]*10), file=fx10)
    #col = []
    #colors = ['#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#228B22']
    #for i in range(0, len(ids_list)):
    #    col.append(colors[ids_list[i]]
    return len(ids_new)

def plot_data_cluster(in_path):
    col_use = ['ID', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    data = pd.DataFrame(csv)
    fig1 = plt.figure(1, figsize=(6, 4))

    colors = ['b', 'g', 'r', 'orange']
    Label_Com = [2, 4,8,11]
    for i,id in enumerate(Label_Com):
        data_pick = data.loc[data['ID'] == id]
        x = np.asarray(data_pick['world_center_x'])
        y = np.asarray(data_pick['world_center_y'])
        print(np.shape(x),np.shape(y))
        plt.scatter(x, y, c=colors[int(i)], cmap='brg', s=40, alpha=0.2, marker='8', linewidth=0)

    plt.ylim(0.01, 0.09)

    ax = fig1.gca()
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.xlabel('x')
    plt.ylabel('y')
    # added this to get the legend to work
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels=Label_Com, loc='upper right')

    plt.show()


def plot_data_cluster1(in_path):  # for original data，
    col_use = ['ID', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    array = np.asarray(csv)
    ids = np.unique(array[:, 0])

    col = [] # 16 个 colors
    colors = ['#FF0000', '#FFA500', '#FFB6C1', '#00FF00', '#228B22','#F08080','#00FA9A',
              '#40E0D0','#FFFF00','#00BFFF','#4169E1', '#48D1CC','#DAA520','#EE82EE','#FF8C00','#8A2BE2']
    for row in array:
        idx = np.where(ids == row[0]) # idx is from 0 to 15
        col.append(colors[np.squeeze(idx)])

    df = pd.DataFrame(csv)
    df.plot.scatter('world_center_x', 'world_center_y', c=col, colormap='jet')

    plt.title('The original data cluster')
    plt.show()
    return


def plot_data_clusterx10(in_path):  # for data x10, x axis -1500 until 100, x 间隔50。 y_axis -250 until +250, y间隔10
    col_use = ['ID', 'world_center_x', 'world_center_y']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')
    array = np.asarray(csv)
    ids = np.unique(array[:, 0])

    col = [] # 16 个 colors
    colors = ['#FF0000', '#FFA500', '#FFB6C1', '#00FF00', '#228B22','#F08080','#00FA9A',
              '#40E0D0','#FFFF00','#00BFFF','#4169E1', '#48D1CC','#DAA520','#EE82EE','#FF8C00','#8A2BE2']
    for row in array:
        idx = np.where(ids == row[0]) # idx is from 0 to 15
        col.append(colors[np.squeeze(idx)])

    df = pd.DataFrame(csv)
    df.plot.scatter('world_center_x', 'world_center_y', c=col, colormap='jet')

    plt.title('The scaled data clusters ')
    x_major_locator = MultipleLocator(50)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(-1500,100)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-250, 250)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.show()
    return


if __name__ == '__main__':
    id_sel_f728_less = np.array([106,107,109,123,121,115,117,119,110,111,113,112], dtype=np.int)
    id_sel_l1 = np.array([59,51,4,48,2,77,11,8,22,12,80,49,134,52])
    id_sel_l2 = np.array([106, 107, 109, 111, 110, 113, 114,115,117,119,118,120,121,123])
    id_sel_senkrecht = np.array([134,107,111,120,114,138,136])
    id_sel_l3 = np.array([2,4,8,11,14,16,17], dtype=np.int)

    #process_detections_result_x10_more_tracks_f100(meas_path_in, out_path_x10_f100, id_sel_senkrecht)
    #process_detections_result_x10_more_tracks_f100(meas_path_in, out_path_x10_f100, id_sel_l1)
    #process_detections_result_x10_all_tracks(meas_path_in, out_path_x10_2)
    #process_detections_result_x10(meas_path_in, meas_path_out_x10)

    #process_detections_result_x10_more_tracks_f728_f950(meas_path_in, out_path_x10_f728_less, id_sel_f728_less)

    l_all_length, count_ids, length_greater_than_80,l_40_80,l_less_40 = get_max_shelter_length(meas_path_in, out_path_max_shelter)
    #for i in range(1,20):
      #  print(all_length[-i])
    print("unique ID 数目： ", count_ids) # 2005
    print("大于80的", length_greater_than_80) # 52
    print("less than80, greater than 40 ", l_40_80)  #
    print("less than 40 ", l_less_40)  #
    print("in total all gaps greater than 2 ", l_all_length)
    # 统计所有大于50的，所有小于70的，所有大于70的是过长的detector导致的gaps,
    #l = create_test_file(meas_path_in)
    #print(l)
    in_ = './dataset/01_TrackTraining_first50frames_for_data_cluster_x10.csv'
    #plot_data_clusterx10(in_)