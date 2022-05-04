import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from .murty import murty_wrapper
from .gibbs_sampler import gibbs_sampler

float_precision = 'f4'


@dataclass (frozen=True)
class TrackerParameters():
    """
    Class containing the overall tracker parameters
    """
    sim_start_ts: int = 728  # start frame of simulation timesteps
    sim_length: int = 946  # 946
    bbox_w: int = 25
    bbox_h: int = 15
    dim_x: int = 4                # Dimension (number) of states
    dim_z: int = 2                # Dimension (number) of measurement inputs
    n_targets_max: int = 1000     # maximum number of targets
    n_gm_cmpnts_max: int = 20     # maximum number of Gaussian mixture components
    log_w_prun_th: float = np.log(0.1)  # -2.3  # Log-likelihood threshold of gaussian mixture weight for pruning # 0.1
    mc_merging_dist_th: float = 3 # Merging threshold of squared Mahalanobis distance between two mixture components
    log_r_sel_th: float = np.log(0.4) # Log-likelihood threshold of target existence probability for selection
    p_survival: float = 0.99      # survival probability
    p_birth: float = 0.7          # birth probability
    adaptive_birth_th: float = 0.05 # adaptive birth threshold
    p_detect: float = 0.9       # detection probability
    log_p_detect: float = field(init=False)
    log_q_detect: float = field(init=False)
    kappa: float = 0.01         # clutter intensity
    log_kappa: float = field(init=False)
    r_prun_th: float = 0.05      # existence probability pruning threshold, original 0.05, faster detect occluded objects and delete it
    log_r_prun_th: float = field(init=False)
    arc_allowable_error: int = 35  # 3.5m
    straight_lane_allowable_error: int = 20  # 2m 
    max_covering_frames: int = 80

    # observation noise covariance
    R: np.ndarray = np.asarray([[1., 0.],
                                [0., 1.]], dtype=float_precision)
    # process noise covariance
    Q: np.ndarray = np.asarray([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [1., 0., 1., 0.],
                                [0., 1., 0., 1.]], dtype=float_precision)
    # Motion model: state transition matrix  # nearly constant vel, zero-mean white-noise acceleration
    F: np.ndarray = np.asarray([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]], dtype=float_precision)
    # Observation model
    H: np.ndarray = np.asarray([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], dtype=float_precision)
    # Initial state covariance matrix
    P_init: np.ndarray = np.asarray([[2., 0., 0., 0.],
                                    [0., 2., 0., 0.],
                                    [0., 0., 2., 0.],
                                    [0., 0., 0., 2.]], dtype=float_precision)
    # Algorithm used for solving the ranked assignment problem
    ranked_assign: Callable[[np.ndarray, np.ndarray, int], None] = murty_wrapper
    num_assignments: int = 1000 # Maximum number of hypothetical assignments created by the ranked assignment

    # Gibbs sampler parameters
    num_samples: int = 1000  # Number of samples the Gibbs sampler takes from the eta_nll matrix
    max_invalid_samples: int = 100 # Maximum number of consecutive invalid samples that do not contain a valid assignment after that the gibbs sampler terminates

    def __post_init__(self):
        """
        Initialization of computed attributes
        """
        object.__setattr__(self, 'log_p_detect', np.log(self.p_detect))
        object.__setattr__(self, 'log_q_detect', np.log(1 - self.p_detect))
        object.__setattr__(self, 'log_kappa', np.log(self.kappa))
        object.__setattr__(self, 'log_r_prun_th', np.log(self.r_prun_th))


@dataclass (frozen=True) 
class SimParameters():
    """
    Class containing the overall simulation parameters
    """
    sim_start_ts: int = 728  # start frame of simulation timesteps
    sim_length: int = 946  # end frame of simulation timesteps  # 946
    dim_x: int = 4   # Dimension (number) of state variables
    dim_z: int = 2   # Dimension of measured state variables
    sigma: float = 0  # Standard deviation of measurement noise
    max_d2: int = 1000**2    # Maximum squared euclidian distance for which py-motmetrics
                            # creates a hypothesis between a ground truth track and an estimated track

    csv_file: str = './dataset/01_TrackTraining_mod_world_x10_more_tracks_f728_f944_less.csv'
    vid_in_file: str = './dataset/vid_in/DJI_in.mp4'
    vid_out_fol: str = './dataset/vid_out'
    tracklets_out_fol: str = './dataset/lmb_tracklets_out'
    # State Transition matrix
    F: np.ndarray = np.asarray([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=float_precision)
    # State Transition matrix for not continuous trajectories
    F_s: np.ndarray = np.asarray([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=float_precision)

    # Data type of array to generate tracks
    dt_init_track_info: np.dtype = np.dtype([('x', 'f8', (dim_x)),  # 4
                                     ('birth_ts', 'u4'),
                                     ('death_ts', 'u4'),
                                     ('label', 'f4')])
    # Data type of tracks
    dt_tracks: np.dtype = np.dtype([('x', 'f8', (dim_x)),
                          ('ts', 'u4'),
                          ('label', 'f4'),
                          ('r', 'f4')])

    # Data type of measurements
    dt_measurement: np.dtype = np.dtype([('z', 'f8', (dim_z)),
                                         ('id', 'u4'),
                                         ('ts', 'u4')])
    v: np.asarray = np.asarray([[-6, 2], [6, 0], [4.4, -0.6], [6.5, 0.5]], dtype='f8')

