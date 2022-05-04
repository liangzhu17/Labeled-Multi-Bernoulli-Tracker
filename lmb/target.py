from copy import deepcopy
import numpy as np
import os
from scipy.special import logsumexp

pdf_out_fol = '/home/liang/lmb-tracker-modified/dataset/plot_pdf_data'
def mkdirs(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def write_plot_result(pdf_out_fol, frame_num, target_label, z_id_list, plt_xy, mlpdf):
    # this frame born target not included
    fol = os.path.join(pdf_out_fol + "/frame_" + str(frame_num))
    mkdirs(str(fol))
    path = os.path.join(fol + "/target_" + str(target_label) + ".csv")
    f = open(path, 'w')
    for i, xy in enumerate(plt_xy):
        print('%d,%d,%.2f,%.2f,%.2f' % (
            int(target_label), int(z_id_list[i]), xy[0], xy[1], mlpdf[i]), file=f)


class Target():
    """
    Represents a single target

    A target is represented by its label, an existence probability, and 
    a probability density function describing the current target state.
    """
    def __init__(self, ts_born, is_good_flag, label, is_target_covered, log_r=0.0, posxy_at_birth=None,z_id=None, pdf=None):
        self.label = int(label)
        self.log_r = log_r
        self.born_frame = ts_born  # the frame where this target is added to self.targets
        self.is_good_flag = is_good_flag  # distinguish targets with high birth_prob (real targets) and low birth_prob (fake)
        self.cover_ts_begin = np.array([-1], dtype=np.int)  # last frame before covering
        self.enter_vel_magnitude = np.empty(dtype=np.float, shape=(1,))
        self.enter_traj_factor = np.empty(dtype=np.float, shape=(2,))
        self.enter_A_pos = np.empty(dtype=np.float, shape=(2,))
        self.point_before_A = np.empty(dtype=np.float, shape=(2,))
        self.v_Ax = np.empty(dtype=np.float, shape=(1,))  # driving direction
        self.vertical_line_through_A = np.empty(dtype=np.float, shape=(2,))

        self.pdf = pdf
        self.z_id = z_id   # The time step when born
        self.z_id_list = []
        self.posxy_at_birth = posxy_at_birth if posxy_at_birth is not None else [0.0,0.0]

        self.leave_B_pos = np.empty(dtype=np.float, shape=(2,))
        self.leave_traj_factor = np.array([None, None], 'f')
        self.leave_vel_magnitude = np.empty(dtype=np.float, shape=(1,))
        self.v_Bx = np.empty(dtype=np.float, shape=(1,))  # driving direction
        self.ts_since_born = 0  # the surviving frames of this target in self.targets since born
        self.is_target_covered = is_target_covered # include messluecke, default is false
        self.ismatched = False

    def predict(self, log_p_survival):
        """
        Predict the state to the next time step

        Parameters
        ----------
        log_p_survival: log of survival probability
        """
        # Predict PDF
        self.pdf.predict()
        # Predict existence probability r, -0.01
        self.log_r += log_p_survival

    def correct(self, assignment_weights):
        """
        Correct the target state based on the computed measurement associations

        Updates the existence probability and combines the association PDFs
        into the resulting PDF. 

        Parameters
        ----------
        assignment_weights : array_like (len(self.assignments))
            Computed hypothesis weights (in log-likelihood) from ranked assignment
        """
        idx = np.where(assignment_weights == 0)[0]  #[0] --> (array([1, 2, 3, 4, 5]),) to [1 2 3 4 5]
        for id in idx:
            assignment_weights[id] = 1e-8

        # 1.: self.log_r = sum of assignment weights, sum of one row of hyp_weights for one target
        self.log_r = np.log(sum(assignment_weights))
        # 2.: Combine PDFs, self.assignments are multiple PDFs
        self.pdf.overwrite_with_merged_pdf(self.assignments, np.log(assignment_weights) - self.log_r)
        # 3.: Reduce complexity of PDF
        self.pdf.reduce_pdf()

    def create_assignments(self, l, Z, ts):
        """
        Compute new hypothetical target-measurement associations

        Parameters
        ----------
        Z : array_like
            measurements
        
        Returns
        -------
        out : array_like (len(z))
            Negative log-likelihood of computed etas (weights) of all target-measurement associations
        """
        # Compute PDFs and weights for each association and a missed detection
        # TODO implementation using array with known size as optimization
        self.assignments = []
        self.z_id_list = []
        for z in Z: 
            # update weights, gm class self.log_w
            self.assignments.append(deepcopy(self.pdf).correct(z['z'], z['id']))
            self.z_id_list.append(z['id'])

        self.assignments.append(deepcopy(self.pdf).correct(None, None))

        # Calculate etas by adding self.log_r
        nll_etas = [- (self.log_r + pdf.log_eta_z) for pdf in self.assignments]

        for i, pdf in enumerate(self.assignments):
            if i < len(self.assignments)-1:
                if str(pdf.z_id) != str(self.z_id_list[i]):
                    print("pdf assignment has different z_id sequence as self.z_id_list , timestep ", ts, "target ", l)
        return self.z_id_list, nll_etas

    def nll_false(self):
        """
        Negative log-likelihood of target being false (died or not born)
        """
        return -np.log(1 - np.exp(self.log_r))

    def __repr__(self):
        """
        String representation of object
        """
        return "\nTarget {}: r={})".format(self.label, np.exp(self.log_r))

