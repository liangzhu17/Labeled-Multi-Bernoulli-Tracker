from operator import attrgetter

from .parameters import TrackerParameters
from .target import Target
from .gm import GM
from .utils import esf
import sqlite3
from sqlite3 import Error
import json
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
from numpy import *
import numpy as np
import os
import math
import warnings
from sklearn import linear_model
label_factor = 10000


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        ## here the file is deleted, to create an empty one for each connection
        print("create db file:")
        if os.path.exists(db_file):
            os.remove(db_file)
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
        exit(1)


conn = create_connection("test.log.sqlite")

create_stmt = open('./dataset/sql_data/tracking_log_db.sql', 'r').read()
c = conn.cursor()  # cursor execute mehtod can run sql command
c.executescript(create_stmt)

# world coordinate tables
insert_world_pos_points_sql = "INSERT INTO world_pos_points(det_id,frame,x,y) Values(?,?,?,?)"
insert_detection_sql = "INSERT OR IGNORE INTO detections(id,frame) Values(?,?)"

# match tables
insert_match_result_sql = "INSERT OR IGNORE INTO detection_match_result (source, target, is_winner, score, score_is_costs, stats) Values(?,?,?,?,?,?)"
# after select, winner's score will be set 1
update_detection_match_sql = "UPDATE detection_match_result SET is_winner = ? WHERE source = ? AND target = ?"
del_track_sql = "DELETE FROM tracks WHERE id=?"

# track tables
insert_track_sql = "INSERT OR IGNORE INTO tracks(id,is_winner,stats) Values(?,?,?)"
insert_track_entry_sql = "INSERT OR IGNORE INTO track_entries(track_id,frame,det_id, is_interpolated, stats) Values(?,?,?,?,?)"
insert_track_pred_sql = "INSERT OR IGNORE INTO track_prediction(track_id,frame,prediction) Values(?,?,?)"
del_track_prediction_sql = "DELETE FROM track_prediction WHERE track_id=? AND frame=?"
sel_source_track_entry_sql = "SELECT det_id FROM track_entries WHERE track_id = ? AND frame = ?"


# to add the gap pair: get the last frame of the target, when he's still alive, this det_id will be the source
sel_cov_target_track_entry_sql = "SELECT max(frame), det_id FROM track_entries WHERE track_id=?"
# get the first frame of the born target, this det_id will be the target
sel_born_target_track_entry_sql = "SELECT min(frame), det_id FROM track_entries WHERE track_id=?"
# update track ids after a match
update_track_entries_tid = "UPDATE track_entries SET track_id = ? WHERE track_id = ?"


# not used :
insert_world_pos_poly_sql = "INSERT OR IGNORE INTO world_pos_polygons(det_id,frame,x1,y1,x2,y2,x3,y3,x4,y4) Values(?,?,?,?,?,?,?,?,?,?)"
insert_screen_pos_bboxes_sql = "INSERT OR IGNORE INTO screen_pos_bboxes(det_id,frame,x,y,width,height) Values(?,?,?,?,?,?)"
insert_img_sql = "INSERT OR IGNORE INTO images(det_id, data) Values(?,?)"
insert_tracklets_sql = "INSERT OR IGNORE INTO tracklets(id) Values(?)"
insert_tracklet_entries_sql = "INSERT OR IGNORE INTO tracklet_entries(tracklet_id,frame,det_id) Values(?,?,?)"
insert_screen_points_pos_sql = "INSERT OR IGNORE INTO screen_pos_points(det_id,frame,x,y) Values(?,?,?,?)"
# delete the not win track prediction and det_match result
sel_target_track_entry_sql = "SELECT track_id FROM track_entries WHERE det_id = ? AND frame = ?"
# check if source target already exists
sel_match_with_s_t = "SELECT * FROM detection_match_result WHERE source = ? AND target = ?"
# select 2 max frames where gap exists
sel_track_entry_2_max_f_det_id = "SELECT frame, det_id FROM track_entries WHERE track_id=? ORDER BY frame DESC LIMIT 2"

c = conn.cursor()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def cal_points_distance(x1,x2,y1,y2):
    return np.sqrt(np.square(x1-x2) + np.square(y1-y2))


def linear_regression_predict(x, y):
    lr = linear_model.LinearRegression()
    lr.fit(x, y)
    return lr


class LMB():
    """
    Main class of the labeled multi bernoulli filter implementation.

    Parameters
    ----------
    params : TrackerParameter, optional
        Parameter object containing all tracker parameters required by all subclasses.
        Gets initialized with default parameters, in case no object is passed.

    LMB-Class specific Parameters
    ------
    params.n_targets_max : int
        Maximum number of targets
    params.log_p_survival : float
        Target survival probability as log-likelihood
    params.p_birth : float
        Maximum birth probability of targets
    params.adaptive_birth_th : float
        Birth probability threshold for existence probability of targets
    params.log_r_prun_th : float
        Log-likelihood threshold of target existence probability for pruning
    self.params.log_r_sel_th : float
        Log-likelihood threshold of target existence probability for selection
    self.params.num_assignments : int
        Maximum number of hypothetical assignments created by the ranked assignment
    dtype_extract : numpy dtype
        Dtype of the extracted targets

    Attributes
    ----------
    targets : array_like
        List of currently active targets
    """

    def __init__(self, params=None):
        self.params = params if params else TrackerParameters()
        self.sim_start_ts = self.params.sim_start_ts
        self._ts = self.sim_start_ts-1
        self.n_targets_max = self.params.n_targets_max
        self.log_p_survival = np.log(self.params.p_survival)
        self.p_birth = self.params.p_birth
        self.adaptive_birth_th = self.params.adaptive_birth_th
        self.log_r_prun_th = self.params.log_r_prun_th
        self.log_r_sel_th = self.params.log_r_sel_th
        self.ranked_assign = self.params.ranked_assign
        self.ranked_assign_num = self.params.num_assignments
        self.dtype_extract = np.dtype([('x', 'f4', self.params.dim_x),
                                       ('P', 'f4', (self.params.dim_x, self.params.dim_x)),
                                       ('r','f4'),
                                       ('label', 'f4'),
                                       ('ts', 'f4'),
                                       ('id', 'f4'),
                                       ('z_id', 'f4')])
        self.targets = []  # list of currently tracked targets
        self.targets_past = []
        self.target_idx_create = []
        self.target_idx_sel = []
        self.target_idx_del = set()
        self.hyp_idx_del = []
        self.targets_new_born_control = []
        self.targets_covered = []
        self.targets_born_to_match = []
        self.tracklets = []
        self.t_c_label_list = []
        self.connect_list = []

    def update(self, z):
        """
        Main function to update the internal tracker state with a measurement

        Parameters
        ----------
        z: measurement object (class to be implemented)

        Returns
        -------
        out: ndarray
            updated and extracted targets of the format : np.dtype([('x', 'f4', dim_x),
                                                                    ('P', 'f4', (dim_x, dim_x)),
                                                                    ('r','f4'),
                                                                    ('label', 'f4'),
                                                                    ('ts','f4')])
        """
        self._ts += 1
        self._z = z
        print('Update step ', self._ts)
        for det in self._z:
            c.execute(insert_world_pos_points_sql, (int(det['id']), self._ts, float(det['z'][0]) / 10, float(det['z'][1]) / 10))
        conn.commit()
        if self._ts == self.sim_start_ts:   # no matches in table
            for zz in self._z:
                c.execute(insert_detection_sql, (int(zz['id']), 0))
            conn.commit()

        self.predict()
        self.correct(z)
        return self.extract(self._select())

    def predict(self):
        """
        Prediction step of LMB tracker

        Predicts states and existence probabilities of every currently tracked target
        """
        # add the long survived targets into self.target_past and compare their log_r
        for target in self.targets:
            target.predict(self.log_p_survival)  # in predict logr = log_r -0.01
        print('Predicted targets ', self.targets)

    def correct(self, z):
        """
        Correction step of LMB tracker

        Correct the predicted track states and their existence probabilities using the new measurement

        Parameters
        ----------
        z: measurement object (class to be implemented)
        """
        ## 1. Create target-measurement associations and calculate each log_likelihood
        ## create association cost matrix with first column for death and second for missed detection
        ## (initialize with min prob --> high negative value due to log):
        N = len(self.targets)
        M = len(z['z'])
        self.C = np.zeros((N, 2 + M))
        self.z_idx_ordered = []
        targets_no_measure = []
        if N > 0:
            C = np.zeros((N, 2 + M))
            ## compute entries of cost matrix for each target-measurement association (including misdetection)
            filter_short_tc_track_label = []
            # the label of targets, those get no measurement this frame
            for i, target in enumerate(self.targets):
                self.target_idx_create.append(target.label)  # its index is i
                # associations and missed detection (second-last column), range(8) is [0,8]
                # create assignment return log_eta_z list for each target, its z1...to z_M
                self.z_idx_ordered, C[i, range(M + 1)] = target.create_assignments(target.label, z, self._ts)
                # died (last column)
                C[i, (M + 1)] = target.nll_false()
                det_id = np.argmin(C[i, range(M + 1)])
                if target.is_good_flag and det_id >= M and target.label not in self.t_c_label_list:  # find targets with measurement gap, avoid adding repeat items
                    # check back, find most recent state of this target with one mc
                    targets_no_measure.append(target.label)
                    last_point = [t_past for t_past in self.targets_past if int(t_past[1]) == int(target.label) and t_past[0] == self._ts-1]
                    sel_last_point = last_point[-1]
                    # assume here at least two frames positions for one missed target, can be less than or equal to 10 frames
                    sel_mc = [t_past[3] for t_past in self.targets_past if
                             int(t_past[1]) == int(target.label) and t_past[0] <= self._ts-2 and t_past[0] >= self._ts-11]
                    sel_frames = [t_past[0] for t_past in self.targets_past if
                             int(t_past[1]) == int(target.label) and t_past[0] <= self._ts-2 and t_past[0] >= self._ts-11]
                    if len(sel_mc)<5:  # this track only survives one frame, delete it
                        filter_short_tc_track_label.append(target.label)
                        continue

                    first_point = [t_past for t_past in self.targets_past if int(t_past[1]) == int(target.label) and t_past[0] == min(sel_frames)]
                    sel_first_point = first_point[-1]
                    dist_first_last_point = np.sqrt(
                        np.square(np.squeeze(sel_last_point[-1]['x'])[0] - np.squeeze(sel_first_point[-1]['x'])[0])
                        + np.square(np.squeeze(sel_last_point[-1]['x'])[1] - np.squeeze(sel_first_point[-1]['x'])[1]))
                    mean_vel = dist_first_last_point / (sel_last_point[0] - sel_first_point[0])
                    point_before_A = [t_past for t_past in self.targets_past if int(t_past[1]) == int(target.label) and t_past[0]==max(sel_frames)]
                    sel_point_before_A = point_before_A[-1]
                    x_enter = [[np.squeeze(mc['x'])[0]] for mc in sel_mc]
                    y_enter = [[np.squeeze(mc['x'])[1]] for mc in sel_mc]
                    lr_enter = linear_regression_predict(x_enter, y_enter)
                    lr_intercept = np.squeeze(sel_last_point[-1]['x'])[1] - np.squeeze(lr_enter.coef_) * np.squeeze(sel_last_point[-1]['x'])[0]  # b=y-kx

                    target_pick = Target(target.born_frame, True, sel_last_point[1], True, log_r=sel_last_point[2], z_id=target.z_id,
                                         pdf=GM(params=self.params, x0=None, prior_mc=sel_last_point[-1]))
                    target_pick.enter_A_pos = np.array([np.squeeze(sel_last_point[-1]['x'])[0], np.squeeze(sel_last_point[-1]['x'])[1]],'f')
                    target_pick.cover_ts_begin = self._ts-1
                    target_pick.v_Ax = np.array([(np.squeeze(sel_last_point[-1]['x'])[0] - np.squeeze(sel_first_point[-1]['x'])[0]) / (
                                 sel_last_point[0] - sel_first_point[0])], 'f')  # vx
                    target_pick.enter_traj_factor = np.array([lr_enter.coef_, lr_intercept], 'f')
                    target_pick.enter_vel_magnitude = np.array([mean_vel],'f')
                    target_pick.point_before_A = np.array([np.squeeze(sel_point_before_A[-1]['x'])[0],np.squeeze(sel_point_before_A[-1]['x'])[1]],'f')

                    target_pick.vertical_line_through_A[0]=-1/lr_enter.coef_ if lr_enter.coef_ != 0.0 else None
                    if lr_enter.coef_ != 0.0:
                        target_pick.vertical_line_through_A[1] = target_pick.enter_A_pos[1]-(-1/lr_enter.coef_)*target_pick.enter_A_pos[0]

                    self.targets_covered.append(target_pick)  # find the covered target

                predictions_dict = OrderedDict()
                for j, assign in enumerate(target.assignments):  # j max len()-1
                    if j < len(target.assignments) - 1:  # exclude last assignment, which is for missed detection
                        predictions_dict[str(assign.z_id)] = {"m":  (np.squeeze(assign.mc['x'])[:2]/10).tolist(),
                                                              "cm": (np.squeeze(assign.mc['P'])[:2, :2]).tolist()}
                        c.execute(sel_source_track_entry_sql, (int(target.label), self._ts - 1))
                        source = c.fetchall()
                        # add assign.mc['log_w'][0] as information {}
                        if len(source) == 1:
                            source = np.squeeze(source)
                            # detection match result only stores won targets' source from last frame
                            if int(assign.z_id) != self.z_idx_ordered[j]:
                                print("assignment zid not same as z index in Matrix C ", assign.z_id, " and ", self.z_idx_ordered[j])
                            c.execute(insert_match_result_sql,
                                      (int(source), int(assign.z_id), 0, C[i, j], 1, "{}"))
                        conn.commit()
                match_information = {
                    "type": "LMBPrediction",
                    "predictions": predictions_dict
                }
                match_info_json = json.dumps(match_information, cls=NumpyEncoder)
                # insert track predictions for one track
                c.execute(insert_track_pred_sql, (int(target.label), self._ts, match_info_json))
                conn.commit()

            t_list = []
            self.t_c_label_list = []
            for t in self.targets_covered:
                if self._ts-t.cover_ts_begin < self.params.max_covering_frames and t.label not in filter_short_tc_track_label:
                    t_list.append(t)
                    self.t_c_label_list.append(t.label)
            self.targets_covered = t_list
            # check the time since covering and time since unmatched     

            # not limit the covered target position, count missing_time
            self.C = C
            ## Ranked assignment
            ## 2. Compute hypothesis weights using specified ranked assignment (murty) algorithm
            hyp_weights = np.zeros((N, M + 2))
            # if non-square matrix，must set extend_cost=True, automatic changed to square matrix
            self.ranked_assign(C, hyp_weights, self.ranked_assign_num)
            ## 3. Calculate resulting existence probability of each target
            for i, target in enumerate(self.targets):
                # get one row, one target to all measurments one to one matchprob,sum is exist_prob,last 2 column is miss-det,correct(None)
                target.correct(
                    hyp_weights[i, :-1])  # only modify PDF， use hyp_weights to calculates log_w,gaussian mixture's weights，not update hyp_weights matrix
        else:
            # No tracked targets yet, set all assignment weights to 0
            hyp_weights = np.zeros((1, M + 2))

        self._adaptive_birth(z, hyp_weights[:, :-2])
        ## 4. Prune targets
        self._prune()
        # delete from self.targets
        self.targets = [t for t in self.targets if t.label not in targets_no_measure]

        for target in self.targets:
            if target.is_good_flag:
                self.targets_past.append([self._ts, target.label, target.log_r, target.pdf.mc])
        # update the self.target_past list, to store 11 past frames include current frame，always able to find targets with one mc state
        # update the self.targets_past for t_covered and t_born???
        help_list = []
        for t in self.targets_past:
            t_b_l = [t_b for t_b in self.targets_born_to_match if t_b.label == t[1]]
            if len(t_b_l) != 0 or t[1] in self.t_c_label_list:
                help_list.append(t)
            else:
                if t[0] > self._ts - 12:  # other targets not covered and not in born_to_match choose most recent 11 frames
                    help_list.append(t)
        self.targets_past = help_list
        print('Corrected, born, and pruned targets ', self.targets)
        filter_short_track_label = []
        # calculate the lines for born targets to be matched, a target can be remained as before or being matched
        # REDEFINE the velocity of t_b. mean vel = (dist_point2_point1)/(f2-f1)
        for i, t_b in enumerate(self.targets_born_to_match):  # not all targets_born_to_match gets a line in this frame

            # a long one wait for 10 points or a short tracklet with many measurement gaps
            if (t_b.label in self.t_c_label_list) or (t_b.label not in self.t_c_label_list and self._ts-t_b.born_frame == 10): # for ten points: # if length of mc is not one ???
                # Point B , last ten or less than ten points will be chosen <=  <=
                sel_first_point = [t_past for t_past in self.targets_past if int(t_past[1]) == int(t_b.label) and t_past[0]==t_b.born_frame]
                first_point = sel_first_point[-1]
                mcs_b = [t_past[3] for t_past in self.targets_past if int(t_past[1]) == int(t_b.label) and t_past[0]>=(t_b.born_frame+1) and t_past[0]<=(t_b.born_frame+10)]
                if len(mcs_b)<5:  # find one frame track or less than 5 frames, delete it
                    filter_short_track_label.append(t_b.label)
                    continue
                sel_frames = [t_past[0] for t_past in self.targets_past if int(t_past[1]) == int(t_b.label) and t_past[0]>=(t_b.born_frame+1) and t_past[0]<=(t_b.born_frame+10)]
                sel_last_point = [t_past for t_past in self.targets_past if int(t_past[1]) == int(t_b.label) and t_past[0] == max(sel_frames)]
                last_point = sel_last_point[-1]
                dist_first_last_point = np.sqrt(np.square(np.squeeze(last_point[-1]['x'])[0]-np.squeeze(first_point[-1]['x'])[0])
                                            + np.square(np.squeeze(last_point[-1]['x'])[1]-np.squeeze(first_point[-1]['x'])[1]))
                mean_vel = dist_first_last_point/(last_point[0] - first_point[0])
                x_leave = [[np.squeeze(mc['x'])[0]] for mc in mcs_b]
                y_leave = [[np.squeeze(mc['x'])[1]] for mc in mcs_b]
                lr_leave = linear_regression_predict(x_leave, y_leave)
                lr_leave_intercept = np.squeeze(first_point[-1]['x'])[1] - np.squeeze(lr_leave.coef_) * \
                                     np.squeeze(first_point[-1]['x'])[0]  # b=y-kx
                t_b.leave_B_pos = np.array([np.squeeze(first_point[-1]['x'])[0], np.squeeze(first_point[-1]['x'])[1]],'f')

                t_b.leave_traj_factor = np.array([lr_leave.coef_, lr_leave_intercept], 'f')
                t_b.leave_vel_magnitude = np.array([mean_vel], 'f')
                t_b.v_Bx = np.array([(np.squeeze(last_point[-1]['x'])[0]-np.squeeze(first_point[-1]['x'])[0])/(last_point[0] - first_point[0])],'f')
                
                cost_line_list = []
                cost_curve_list = []                
                
                # match starts from here
                for i,t_c in enumerate(self.targets_covered):                   
                    
                    # straight lane, abs(delta_k) < 0.3, -5.7 < alpha < 5.7(degree)
                    delta_k = t_c.enter_traj_factor[0] - t_b.leave_traj_factor[0]
                    if abs(delta_k) < 0.3:
                        dist_AB = cal_points_distance(t_c.enter_A_pos[0],t_b.leave_B_pos[0],t_c.enter_A_pos[1],t_b.leave_B_pos[1])
                        d_mean_vel = (t_b.born_frame - t_c.cover_ts_begin) * (t_c.enter_vel_magnitude + t_b.leave_vel_magnitude) / 2

                        if abs(dist_AB - d_mean_vel) < self.params.straight_lane_allowable_error:  # /0.02 得到 m/s, 距离单位是米，如果1m以内 ??? 考虑一下参数。
                            score = abs(dist_AB-d_mean_vel)
                            # find the match, update the old targets_covered
                            # if t_b started being covered update the t_c traj_params and tables if not only update the tables
                            # both cases are same, target t_b already being covered or not
                            match_info = {"match type": "straight lane",
                                          "slope of line entering shelter": np.float64(t_c.enter_traj_factor[0]),
                                          "slope of line leaving shelter ": np.float64(t_b.leave_traj_factor[0])}
                            match_info_json = json.dumps(match_info, cls=NumpyEncoder)
                            conn_pairs = []
                            for cp in self.connect_list:
                                if cp[1] == t_b.label:
                                    conn_pairs = cp
                            if len(conn_pairs)!=0: # if this is not a first time connection for t_c
                                c.execute(sel_cov_target_track_entry_sql, (conn_pairs[0]))
                                sel_source = c.fetchall()
                                if len(sel_source) == 1:
                                    sel_source = np.squeeze(sel_source)
                                    c.execute(insert_match_result_sql,
                                              (int(sel_source[-1]), t_b.z_id, 1, score, 1, match_info_json))
                                    c.execute(update_track_entries_tid, (t_c.label, t_b.label))
                                    c.execute(del_track_sql, (t_b.label,))
                            else:  # if this is first time connection
                                c.execute(sel_cov_target_track_entry_sql, (t_c.label,))
                                sel_source = c.fetchall()
                                if len(sel_source) == 1:
                                    sel_source = np.squeeze(sel_source)
                                    c.execute(insert_match_result_sql, (int(sel_source[-1]), t_b.z_id,1,score,1,match_info_json))
                                    c.execute(update_track_entries_tid, (t_c.label, t_b.label))
                                    c.execute(del_track_sql, (t_b.label,))

                            self.targets_past = [row for row in self.targets_past if row[1]!= t_c.label]
                            self.targets_covered = [t for t in self.targets_covered if t.label != t_c.label]
                            del self.targets_born_to_match[i]
                            self.connect_list.append([t_c.label, t_b.label])
                            self.t_c_label_list = [t.label for t in self.targets_covered]
                            break

                    else:  # curved lane  (Turnround is a exclusion, no need for curve calculation)
                        if (np.isnan(t_c.vertical_line_through_A).any() and (t_c.enter_A_pos[0] -t_c.point_before_A[0]) * (t_c.enter_A_pos[0] - t_b.leave_B_pos[0]) < 0) or (
                                not np.isnan(t_c.vertical_line_through_A).any()
                                and (t_c.point_before_A[1] - t_c.vertical_line_through_A[0] * t_c.point_before_A[0] -
                                     t_c.vertical_line_through_A[1])
                                * (t_b.leave_B_pos[1] - t_c.vertical_line_through_A[0] * t_b.leave_B_pos[0] -
                                   t_c.vertical_line_through_A[1]) < 0):

                            Cx = (t_c.enter_traj_factor[1] - t_b.leave_traj_factor[1]) / (t_b.leave_traj_factor[0] - t_c.enter_traj_factor[0])
                            Cy = Cx * t_c.enter_traj_factor[0] + t_c.enter_traj_factor[1]
                            dist_CA = cal_points_distance(Cx, t_c.enter_A_pos[0], Cy, t_c.enter_A_pos[1])
                            dist_CB = cal_points_distance(Cx, t_b.leave_B_pos[0], Cy, t_b.leave_B_pos[1])
                            vector_BBs = np.array([t_b.v_Bx, t_b.leave_traj_factor[0] * t_b.v_Bx], 'f')
                            vector_CB = np.array([t_b.leave_B_pos[0] - Cx, t_b.leave_B_pos[1] - Cy], 'f')
                            dot_product_CB = vector_CB[0] * vector_BBs[0] + vector_CB[1] * vector_BBs[1]
                            vector_AAs = np.array([t_c.v_Ax, t_c.enter_traj_factor[0] * t_c.v_Ax], 'f')
                            vector_CA = np.array([t_c.enter_A_pos[0] - Cx, t_c.enter_A_pos[1] - Cy], 'f')
                            dot_product_CA = vector_AAs[0] * vector_CA[0] + vector_AAs[1] * vector_CA[1]
                            # if slope not exists and slope exists, except the turnaround case

                            # to angle formula line_enter to line_leave, line_enter to angle bisector
                            tan_AB = (t_b.leave_traj_factor[0] - t_c.enter_traj_factor[0]) / (
                                            1 + t_c.enter_traj_factor[0] * t_b.leave_traj_factor[0])
                            AtoB_angle = np.arctan(tan_AB)+math.pi if np.arctan(tan_AB)<0 else np.arctan(tan_AB) # np.arctan res [-0.5pi,0.5pi], maps it to [0,pi]

                            tan_half_theta = np.tan(0.5 * AtoB_angle)
                            line_bisector_coef = (tan_half_theta + t_c.enter_traj_factor[0]) / (
                                        1 - t_c.enter_traj_factor[0] * tan_half_theta)
                            line_bisector_intercept = Cy - line_bisector_coef * Cx  # C ist on the line of angle bisector
                            alpha = np.arccos((vector_CA[0]*vector_CB[0]+vector_CA[1]*vector_CB[1])/(dist_CB*dist_CA))   # ret value is [0,pi]
                            radius = np.tan(0.5*alpha) * (dist_CB + dist_CA) / 2
                            # calculate the circle center o, equation ax=b
                            eq_a = np.array([[line_bisector_coef, -1], [-t_c.enter_traj_factor[0], 1]], 'f')
                            eq_b1 = np.array([-line_bisector_intercept,
                                                  t_c.enter_traj_factor[1] + radius * np.sqrt(
                                                      1 + np.square(t_c.enter_traj_factor[0]))], 'f')
                            eq_b2 = np.array([-line_bisector_intercept,
                                                  t_c.enter_traj_factor[1] - radius * np.sqrt(
                                                      1 + np.square(t_c.enter_traj_factor[0]))], 'f')
                            pos_o1 = np.linalg.solve(eq_a, eq_b1)
                            pos_o2 = np.linalg.solve(eq_a, eq_b2)
                            # compare angle AO1B and AO2B, choose larger one, bogenlänge r*alpha
                            vector_O1A = np.array([t_c.enter_A_pos[0] - pos_o1[0], t_c.enter_A_pos[1] - pos_o1[1]], 'f')
                            vector_O1B = np.array([t_b.leave_B_pos[0] - pos_o1[0], t_b.leave_B_pos[1] - pos_o1[1]], 'f')
                            vector_O2A = np.array([t_c.enter_A_pos[0] - pos_o2[0], t_c.enter_A_pos[1] - pos_o2[1]], 'f')
                            vector_O2B = np.array([t_b.leave_B_pos[0] - pos_o2[0], t_b.leave_B_pos[1] - pos_o2[1]], 'f')
                            dist_O1A = np.sqrt(np.square(pos_o1[0] - t_c.enter_A_pos[0]) + np.square(pos_o1[1] -
                                       t_c.enter_A_pos[1]))
                            dist_O1B = np.sqrt(np.square(t_b.leave_B_pos[0] - pos_o1[0]) + np.square(
                                       t_b.leave_B_pos[1] - pos_o1[1]))
                            dist_O2A = np.sqrt(np.square(t_c.enter_A_pos[0] - pos_o2[0]) + np.square(
                                       t_c.enter_A_pos[1] - pos_o2[1]))
                            dist_O2B = np.sqrt(np.square(t_b.leave_B_pos[0] - pos_o2[0]) + np.square(
                                       t_b.leave_B_pos[1] - pos_o2[1]))
                            cos_AO1B = (vector_O1A[0] * vector_O1B[0] + vector_O1A[1] * vector_O1B[1]) / (
                                        dist_O1A * dist_O1B)
                            cos_AO2B = (vector_O2A[0] * vector_O2B[0] + vector_O2A[1] * vector_O2B[1]) / (
                                        dist_O2A * dist_O2B)
                            arc_length = radius * np.arccos(cos_AO2B) if cos_AO1B >= cos_AO2B else radius * np.arccos(cos_AO1B)
                            # return value is also positive for obtuse angle, decreasing function
                            cal_displacement = 0.5 * (t_c.enter_vel_magnitude + t_b.leave_vel_magnitude) * (
                                    t_b.born_frame - t_c.cover_ts_begin)
                            delta_arc_length = abs(arc_length - cal_displacement)
                            
                            if delta_arc_length < self.params.arc_allowable_error:  # smaller than 3.5 Meter?
                                score = delta_arc_length
                                match_info = {"match type": "curved lane",
                                              "drawn arc_length": np.float64(arc_length),
                                              "calculated displacement in shelter": np.float64(cal_displacement)}
                                match_info_json = json.dumps(match_info, cls=NumpyEncoder)
                                conn_pairs = []
                                for cp in self.connect_list:
                                    if int(cp[1]) == t_c.label:
                                        conn_pairs = cp
                                if len(conn_pairs) != 0:  # if this is not a first time connection for t_c
                                    c.execute(sel_cov_target_track_entry_sql, (conn_pairs[0],))
                                    sel_source = c.fetchall()
                                    if len(sel_source) == 1:
                                        sel_source = np.squeeze(sel_source)
                                        c.execute(insert_match_result_sql,
                                                  (int(sel_source[-1]), t_b.z_id, 1, score, 1, match_info_json))
                                        c.execute(update_track_entries_tid, (t_c.label, t_b.label))
                                        c.execute(del_track_sql, (t_b.label,))
                                else:  # if this is first time connection
                                    c.execute(sel_cov_target_track_entry_sql, (t_c.label,))
                                    sel_source = c.fetchall()
                                    if len(sel_source) == 1:
                                        sel_source = np.squeeze(sel_source)
                                        c.execute(insert_match_result_sql,
                                                  (int(sel_source[-1]), t_b.z_id, 1, score, 1, match_info_json))
                                        c.execute(update_track_entries_tid, (t_c.label, t_b.label))
                                        c.execute(del_track_sql, (t_b.label,))
                                self.targets_past = [row for row in self.targets_past if row[1] != t_c.label]
                                self.targets_covered = [t for t in self.targets_covered if t.label != t_c.label]
                                del self.targets_born_to_match[i]
                                self.connect_list.append([t_c.label, t_b.label])
                                self.t_c_label_list = [t.label for t in self.targets_covered]
                                break
        self.targets_born_to_match = [t for t in self.targets_born_to_match if t.label not in filter_short_track_label]
        if self._ts==self.sim_start_ts:   # add tracks from frame 0 or other start frame
            for target in self.targets:
                pred_dict = OrderedDict()
                c.execute(insert_track_entry_sql, (int(target.label), self._ts, int(target.z_id), 0, "{}"))
                c.execute(insert_track_sql, (int(target.label), 1, "{}"))  # add tracks not won use update here
                pred_dict[str(target.z_id)] = {"m": (np.squeeze(target.pdf.mc['x'])[:2] / 10).tolist(),
                                                "cm": np.squeeze(target.pdf.mc['P'])[:2, :2].tolist()}
                match_information = {
                    "type": "LMBPrediction",
                    "predictions": pred_dict
                    }
                match_info_json = json.dumps(match_information, cls=NumpyEncoder)
                # insert track predictions for one track
                c.execute(insert_track_pred_sql, (int(target.label), self._ts, match_info_json))
            conn.commit()

    def _adaptive_birth(self, Z, assign_weights):
        """
        Adaptive birth of targets based on measurement

        New targets are born at the measurement locations based on the
        assignment probabilities of these measurements: The higher the
        probability of a measurement being assigned to any existing target,
        the lower the birth probability of a new target at this position.
        按概率决定某个量测被分配给已存在轨迹还是生成新轨迹
        The implementation is based on the proposed algorithm in
        S. Reuter et al., "The Labeled Multi-Bernoulli Filter", 2014

        Parameters
        ----------
        Z : array_like
            measurements
        assign_weights : array_like
            Weights of all track-measurement assignments (without missed detection or deaths)
            Shape: num_tracks x num_measurements
            # Shape: when target is [], num_tracks = 1
        """
        # Probability of each measurement being assigned to an existing target
        z_assign_prob = np.sum(assign_weights, axis=0)  # col sum for each z
        not_assigned_sum = sum(1 - z_assign_prob)

        if not_assigned_sum > 1e-9:  # not divided by 0. Every new born track assigned with born probability
            for z, prob in zip(Z, z_assign_prob):  # get each z in Z[]
                # Set lambda_b to the mean cardinality of the birth multi-Bernoulli RFS
                # This results in setting the birth prob to (1 - prob_assign).
                self.lambda_b = not_assigned_sum
                # limit the birth existence probability to the configured p_birth， 论文中adaptive birth model equation 33, 某个z(一列和)没有匹配的概率占所有无匹配权重和的比例就是prob_birth
                prob_birth = np.minimum(self.p_birth, self.lambda_b * (1 - prob) / not_assigned_sum)
                ## Debug output:
                print("sum ", not_assigned_sum, " assign prob ", prob, " prob_birth ", prob_birth)
                # Spawn only new targets which exceed the existence prob threshold, 只有大于r阈值才创建新目标
                if prob_birth > self.adaptive_birth_th:  # 0.05
                    flag = False # True for detection not from the edge
                    if np.log(prob_birth) > -0.51:  # r > 0.6
                        flag = True
                    self._spawn_target(np.log(prob_birth), x0=[z['z'][0], z['z'][1], 0., 0.], z_id=z['id'], flag=flag)

    def _prune(self):
        """
        Pruning of tracks

        Selection according to the configured pruning threshold for the existence probability.
        TODO: limit the remaining tracks to the configured maximum number based on
        descending existence probability.
        """
        # reduce number of targets by check its prun_exis_probability

        self.targets = [t for t in self.targets if t.log_r > self.log_r_prun_th]

    def _select(self):
        """
        Select the most likely targets

        Compute the most likely cardinality (number) of targets and
        select the corresponding number of targets with the highest existence probability.
        The cardinality is obtained by computing the cardinality distribution of
        the multi-Bernoulli RFS and selecting the cardinality with highest probability.

        TODO: the mean cardinality of a multi-Bernoulli RFS can be obtained by sum(r).
        The applicability for the selection step should be investigated.

        Returns
        -------
        out: list
            selected targets
        """
        # exclude the targets born in this update step
        selected_targets = []
        for t in self.targets:
            if t.born_frame != self._ts:
                selected_targets.append(t)

        # selected_targets = [t for t in self.targets if t.label.split('.')[0] != str(self._ts)]
        # get the existence probabilities of the targets
        r = [np.exp(t.log_r) for t in selected_targets]
        r = np.asarray(r)
        # limit the existence probabilities for numerical stability (avoiding divide by 0)
        r = np.minimum(r, 1. - 1e-9)
        r = np.maximum(r, 1e-9)
        # Calculate the cardinality distribution of the multi-Bernoulli RFS， zero targets or 1,...or 7 targets. 8 possibilities in esf()， 集合相乘，笛卡尔积
        cdn_dist = np.prod(1.0 - r) * esf(r / (1. - r))
        est_cdn = np.argmax(cdn_dist)  # get the col number
        num_tracks = min(est_cdn, self.n_targets_max)
        selected_targets.sort(key=attrgetter('log_r'), reverse=True)
        selected_targets = selected_targets[:num_tracks]

        self.targets_new_born_control = []
        self.target_idx_create_sel_idx = []
        if self._ts == self.sim_start_ts:
            selected_targets = self.targets
        else:
            for t in selected_targets:
                self.target_idx_sel.append(t.label)
            for ridx, t_label in enumerate(self.target_idx_create):
                if t_label not in self.target_idx_sel:
                    self.target_idx_del.add(t_label)
                else:
                    self.target_idx_create_sel_idx.append(ridx)
            for target_to_del in self.target_idx_del:  # delete this not win targets in track_prediction, only won targets are needed
                c.execute(del_track_prediction_sql, (target_to_del, self._ts))
            conn.commit()

        for t in selected_targets:
            c.execute(insert_track_sql, (t.label, 1, "{}"))  # add tracks not won use update here
            conn.commit()
        print("Selected targets: ", selected_targets)
        if self._ts != self.sim_start_ts:
            for idx in self.target_idx_create_sel_idx:  # the index of selected targets in self.targets_idx_create(before correct)
                # for one target, get the won z_id
                l = len(self.z_idx_ordered)
                z_idx_win = np.argmin(self.C[int(idx), range(l + 1)])  # include not selected targets
                if z_idx_win < l:  # max l-1, only store tracks not deleted at this time step
                    c.execute(insert_detection_sql, (int(self.z_idx_ordered[z_idx_win]), self._ts))
                    c.execute(insert_track_entry_sql,
                              (self.target_idx_create[idx], self._ts, int(self.z_idx_ordered[z_idx_win]), 0, "{}"))
                    c.execute(sel_source_track_entry_sql, (self.target_idx_create[idx], self._ts - 1))
                    source = c.fetchall()
                    if len(source) == 1:  # update the won match
                        c.execute(update_detection_match_sql, (1, int(np.squeeze(source)), int(self.z_idx_ordered[z_idx_win])))
                    conn.commit()

        self.target_idx_sel.clear()
        self.target_idx_create = []
        self.target_idx_del.clear()

        return selected_targets

    def _spawn_target(self, log_r, x0, z_id, flag):
        """
        Spawn new target instances

        Parameters
        ----------
        log_r : float
            Log likelihood of initial existence probability
        x0 : array_like
            Initial state
        """

        # label = '{}.{}'.format(self._ts, len(self.targets))
        # The frame number when the target born = self._ts-1, name starts with 1000x
        label = '{}'.format((self._ts + 1) * label_factor + len(self.targets))  # 按顺序排， 11 个就排到11号

        # add IDs, 每个目标都有一个MC
        new_target = Target(self._ts, flag, label, False, log_r=log_r, z_id=z_id, pdf=GM(params=self.params, x0=x0))
        self.targets.append(new_target)
        if flag:
            self.targets_new_born_control.append(new_target)
            if len(self.targets_covered) != 0:
                self.targets_born_to_match.append(new_target)

    def extract(self, selected_targets):
        """
        Extract selected targets from the LMB class instance

        Extract the selected targets with their labels, existence probabilities and their states x
        and covariances P of their corresponding most likely gaussian mixture component.

        Parameters
        ----------
        selected_targets : list
            List of class Targets instances

        Returns
        -------
        out: ndarry
            Ndarray of dtype: self.dtype_extract
        """

        extracted_targets = np.zeros(len(selected_targets), dtype=self.dtype_extract)

        for i, target in enumerate(selected_targets):
            mc_extract_ind = np.argmax(target.pdf.mc['log_w'])
            extracted_targets[i]['x'] = target.pdf.mc[mc_extract_ind]['x']
            extracted_targets[i]['P'] = target.pdf.mc[mc_extract_ind]['P']
            extracted_targets[i]['r'] = np.exp(target.log_r)
            extracted_targets[i]['label'] = target.label
            extracted_targets[i]['ts'] = self._ts
            extracted_targets[i]['z_id'] = target.z_id

        if self._ts == self.params.sim_length-1:
            conn.commit()
            c.close()
        return extracted_targets
