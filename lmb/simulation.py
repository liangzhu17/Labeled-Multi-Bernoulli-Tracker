import numpy as np

from .parameters import SimParameters
import pandas as pd

'''world coordinate'''

csv_mod_path = './dataset/01_TrackTraining_mod_world_x10_more_tracks_f728_f944_less.csv'


def csv_mod_reader():
    csv = pd.read_csv(csv_mod_path, header=None, engine='python')
    gt = np.asarray(csv)
    return gt


gt = np.asarray(csv_mod_reader())


def create_target_tracks(params=None):
    """
    Creates target tracks with specified simulation length, initial states, motion model, birth and death timestamps

    Parameters
    ----------
    params: Instance of the SimParameters class

    Returns
    -------
    out: ndarray
        Array of ground truths tracks (dtype = SimParameters.dt_tracks)

    """
    params = params if params else SimParameters()
    gt = pd.read_csv(params.csv_file, header=None, engine='python')
    gt = np.asarray(gt)

    tracks = np.zeros(len(gt), dtype=params.dt_tracks)
    #gt_track_history = np.zeros(0, dtype=params.dt_tracks)

    v = np.array([4.4,-0.6])
    for ridx in range(len(gt)):
        tid = gt[ridx, 1]
        # v = params.v[0, :] if tid == 2 else params.v[1, :]
        tracks['x'][ridx] = np.hstack((gt[ridx, 3:5], v))
        tracks['ts'][ridx] = gt[ridx, 2]
        tracks['label'][ridx] = tid

    return tracks


def create_measurement_history(params=None):
    """
    Adds measurement noise to gt_target_tracks 
    TODO: Reduces gt_target_tracks by missed detections and add clutter

    Parameters
    ----------
    gt_target_track_history: ndarray
        Array of ground truth tracks (dtype = SimParameters.dt_tracks)
                                                    
    params: Instance of the SimParameters class

    Returns
    -------
    out: ndarray
        Array of the measurement history (dtype = SimParameters.dt_measurement)
    """

    params = params if params else SimParameters()

    measurement_history = np.zeros(len(gt), dtype=params.dt_measurement)

    for ridx in range(len(gt)):
        measurement_history['z'][ridx] = gt[ridx, 3:5]
        measurement_history['ts'][ridx] = gt[ridx, 2]
        measurement_history['id'][ridx] = gt[ridx, 0]

    return measurement_history
