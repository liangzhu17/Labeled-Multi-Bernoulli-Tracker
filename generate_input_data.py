import numpy as np
import pandas as pd

'''add row indices and modify file names'''

meas_path_in_iou = './dataset/01_IOUTracks.csv'
meas_path_out_iou = './dataset/01_IOUTracks_with_detid_bildschirm.csv'

meas_path_in = './dataset/01_TrackTraining.csv'
meas_path_out = './dataset/01_TrackTraining_mod.csv'


def write_meas_result(in_path, out_path, iou=True):

    col_use = ['ID', 'frame', 'world_x', 'world_y', 'speed', 'heading'] if iou else ['ID', 'Time','screen_bbox_x1', 'screen_bbox_y1','screen_bbox_x2', 'screen_bbox_y2']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')

    arr = np.asarray(csv)
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    f = open(out_path, 'w')
    for row in res:
        print('%d,%d,%d,%.2f,%.2f,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6]), file=f)  # changed sequence fid,tid,bbox,..


def write_meas_bildschirm_iou_result(in_path, out_path):
    col_use = ['ID', 'Time', 'screen_x','screen_y','screen_width', 'screen_height']
    csv = pd.read_csv(in_path, usecols=col_use, engine='python')

    arr = np.asarray(csv)
    m = np.arange(0, len(arr))
    res = np.insert(arr, 0, values=m, axis=1)
    f = open(out_path, 'w')
    for row in res:
        print('%d,%d,%d,%.2f,%.2f,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3], row[4], row[5], row[6]), file=f)  # changed sequence fid,tid,bbox,..


'''image coordinate  --> get center point, all tracks same velocity, test if makes sense'''


def csv_img_coord_mod_reader(in_path, out_path):
    csv = pd.read_csv(in_path, header=None, engine='python')
    arr = np.asarray(csv)
    f = open(out_path, 'w')
    for row in arr:
        cx = (row[3] + row[5]) / 2
        cy = (row[4] + row[6]) / 2
        row[3] = cx
        row[4] = cy

    for row in arr:
        print('%d,%d,%d,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3], row[4]), file=f)  # changed sequence fid,tid,bbox,..


if __name__ == '__main__':


    #csv_img_coord_mod_reader(meas_path_out, meas_path_out)
    write_meas_bildschirm_iou_result(meas_path_in_iou, meas_path_out_iou)