""" 
A 2D Tracker based on input drone video data and 2D obsevation data.
"""

import lmb
import cv2
import numpy as np
from datetime import datetime
import random
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def mkdirs(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def write_tracklets_result(fol, res):
    mkdirs(str(fol))
    current_date = datetime.now()
    date_time = current_date.strftime("%Y_%m_%d_%H-%M-%S")
    report_name = 'tracklets_{}.csv'.format(date_time)
    path = os.path.join(fol + "/" + str(report_name))
    f = open(path, 'w')
    for r in res:   # label, ts, pos_x, pos_y, vx, vy
        print('%d,%d,%.4f,%.4f,%.4f,%.4f' % (
            int(r[0]), int(r[1]), r[2], r[3], r[4], r[5]), file=f)


def vid_writer(video_path_in, video_path_out):
    dir_out = str(video_path_out)
    mkdirs(dir_out)
    file_out_name = os.path.join(dir_out + "/" + "track_out_0119_test.mp4")
    try:
        cap = cv2.VideoCapture(int(video_path_in))
    except:
        cap = cv2.VideoCapture(video_path_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 3840 x 2160
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(file_out_name, fourcc, fps, (video_width, video_height))
    return out


def main():

    sim_params = lmb.SimParameters() 
    tracker_params = lmb.TrackerParameters()

    tracker = lmb.LMB(params=tracker_params)

    gt_target_track_history = lmb.create_target_tracks(params=sim_params)
    # measurement dim same as tracker_history
    measurement_history = lmb.create_measurement_history(params=sim_params)
    # type ->  x, ts, l, r
    tracker_est_history = np.zeros(0, dtype = sim_params.dt_tracks)

    try:
        vid = cv2.VideoCapture(int(sim_params.vid_in_file))
    except:
        vid = cv2.VideoCapture(sim_params.vid_in_file)

    for ts in range(sim_params.sim_start_ts, sim_params.sim_length):

        vid.set(cv2.CAP_PROP_POS_FRAMES, int(ts))
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        print('Frame #: ', ts, flush=True)
        original_h, original_w, _ = frame.shape

        # Call the tracker
        tracker_est = tracker.update(measurement_history[measurement_history['ts']==ts])

        tracker_est_ts = np.zeros(len(tracker_est), dtype=sim_params.dt_tracks)
        tracker_est_ts['ts'] = ts
        tracker_est_ts['label'] = tracker_est['label']
        tracker_est_ts['x'] = tracker_est['x']
        tracker_est_ts['r'] = tracker_est['r']

        tracker_est_history = np.concatenate((tracker_est_history, tracker_est_ts))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    write_tracklets_result(sim_params.tracklets_out_fol,np.asarray(tracker.tracklets))

    #mot_summary, mot_ts_results = lmb.evaluate_point_2D(gt_target_track_history, tracker_est_history, sim_params.max_d2)
    #lmb.create_report_point_2D(gt_target_track_history, tracker_est_history, mot_summary, mot_ts_results)


if __name__ == '__main__':
    main()
