import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .curves import *
from .macro import *
import copy
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")

def point_distance(p1,p2,type="l2"):
    if type.lower()=="l2":
        return np.sqrt(np.sum((p1-p2)**2))
    elif type.lower()=="l1":
        return np.sum(np.abs(p1-p2))
    else:
        raise NotImplementedError(f"Distance type {type} not yet supported")

def create_matched_pair(list1, list2, row_indices, col_indices):
    """
    Creates a list of matched pairs based on the row and column indices.

    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.
        row_indices (list): List of row indices based on Hungarian Matching
        col_indices (list): List of row indices based on Hungarian Matching

    Returns:
        list: List of matched pairs, where each pair is a list containing an element from list1 and an element from list2.
    """
    assert len(list1) == len(list2)
    assert len(row_indices) == len(col_indices)
    
    matched_pair = []
    for i in range(len(row_indices)):
        matched_pair.append([list1[row_indices[i]], list2[col_indices[i]]])
    
    return matched_pair


##########################   base  ###########################
class SketchBase(object):
    """Base class for sketch (a collection of curves). """
    def __init__(self, children, reorder=True):
        self.children = children

        if reorder:
            self.reorder()

    @staticmethod
    def from_dict(stat):
        """construct sketch from json data

        Args:
            stat (dict): dict from json data
        """
        raise NotImplementedError

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        """construct sketch from vector representation

        Args:
            vec (np.array): (seq_len, n_args)
            start_point (np.array): (2, ). If none, implicitly defined as the last end point.
        """
        raise NotImplementedError

    def reorder(self):
        """rearrange the curves to follow counter-clockwise direction"""
        raise NotImplementedError

    @property
    def start_point(self):
        return self.children[0].start_point

    @property
    def end_point(self):
        return self.children[-1].end_point

    @property
    def bbox(self):
        """compute bounding box (min/max points) of the sketch"""
        all_points = np.concatenate([child.bbox for child in self.children], axis=0)
        return np.stack([np.min(all_points, axis=0), np.max(all_points, axis=0)], axis=0)

    @property
    def bbox_size(self):
        """compute bounding box size (max of height and width)"""
        bbox_min, bbox_max = self.bbox[0], self.bbox[1]
        bbox_size = np.max(np.abs(np.concatenate([bbox_max - self.start_point, bbox_min - self.start_point])))
        return bbox_size

    @property
    def global_trans(self):
        """start point + sketch size (bbox_size)"""
        return np.concatenate([self.start_point, np.array([self.bbox_size])])

    def transform(self, translate, scale):
        """linear transformation"""
        for child in self.children:
            child.transform(translate, scale)

    def flip(self, axis):
        for child in self.children:
            child.flip(axis)
        self.reorder()

    def numericalize(self, n=256):
        """quantize curve parameters into integers"""
        for child in self.children:
            child.numericalize(n)

    def normalize(self, size=256):
        """normalize within the given size, with start_point in the middle center"""
        cur_size = self.bbox_size
        scale = (size / 2 * NORM_FACTOR - 1) / cur_size # prevent potential overflow if data augmentation applied
        self.transform(-self.start_point, scale)
        self.transform(np.array((size / 2, size / 2)), 1)

    def denormalize(self, bbox_size, size=256):
        """inverse procedure of normalize method"""
        scale = bbox_size / (size / 2 * NORM_FACTOR - 1)
        self.transform(-np.array((size / 2, size / 2)), scale)

    def to_vector(self):
        """convert to vector representation"""
        raise NotImplementedError

    def draw(self, ax):
        """draw sketch on matplotlib ax"""
        raise NotImplementedError

    def to_image(self):
        """convert to image"""
        fig, ax = plt.subplots()
        self.draw(ax)
        ax.axis('equal')
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return X

    def sample_points(self, n=32):
        """uniformly sample points from the sketch"""
        raise NotImplementedError


####################### loop & profile #######################
class Loop(SketchBase):
    """Sketch loop, a sequence of connected curves."""
    @staticmethod
    def from_dict(stat):
        all_curves = [construct_curve_from_dict(item) for item in stat['profile_curves']]
        this_loop = Loop(all_curves)
        this_loop.is_outer = stat['is_outer']
        return this_loop

    def __repr__(self):
        s = "Loop:"
        for curve in self.children:
            s += "\n      -" + str(curve)
        return s
    
    def __str__(self):
        return "Loop:" + "\n      -" + "\n      -".join([str(curve) for curve in self.children])

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        all_curves = []
        if start_point is None:
            # FIXME: explicit for loop can be avoided here
            for i in range(vec.shape[0]):
                if vec[i][0] == EOS_IDX:
                    start_point = vec[i - 1][1:3]
                    break
        for i in range(vec.shape[0]):
            type = vec[i][0]
            if type == SOL_IDX:
                continue
            elif type == EOS_IDX:
                break
            else:
                curve = construct_curve_from_vector(vec[i], start_point, is_numerical=is_numerical)
                start_point = vec[i][1:3] # current curve's end_point serves as next curve's start_point
            all_curves.append(curve)
        return Loop(all_curves)

    def reorder(self):
        """reorder by starting left most and counter-clockwise"""
        if len(self.children) <= 1:
            return

        start_curve_idx = -1
        sx, sy = 10000, 10000

        # correct start-end point order
        if np.allclose(self.children[0].start_point, self.children[1].start_point) or \
            np.allclose(self.children[0].start_point, self.children[1].end_point):
            self.children[0].reverse()

        # correct start-end point order and find left-most point
        for i, curve in enumerate(self.children):
            if i < len(self.children) - 1 and np.allclose(curve.end_point, self.children[i + 1].end_point):
                self.children[i + 1].reverse()
            if round(curve.start_point[0], 6) < round(sx, 6) or \
                    (round(curve.start_point[0], 6) == round(sx, 6) and round(curve.start_point[1], 6) < round(sy, 6)):
                start_curve_idx = i
                sx, sy = curve.start_point

        self.children = self.children[start_curve_idx:] + self.children[:start_curve_idx]

        # ensure mostly counter-clock wise
        if isinstance(self.children[0], Circle) or isinstance(self.children[-1], Circle): # FIXME: hard-coded
            return
        start_vec = self.children[0].direction()
        end_vec = self.children[-1].direction(from_start=False)
        if np.cross(end_vec, start_vec) <= 0:
            for curve in self.children:
                curve.reverse()
            self.children.reverse()

    def to_vector(self, max_len=None, add_sol=True, add_eos=True):
        loop_vec = np.stack([curve.to_vector() for curve in self.children], axis=0)
        if add_sol:
            loop_vec = np.concatenate([SOL_VEC[np.newaxis], loop_vec], axis=0)
        if add_eos:
            loop_vec = np.concatenate([loop_vec, EOS_VEC[np.newaxis]], axis=0)
        if max_len is None:
            return loop_vec

        if loop_vec.shape[0] > max_len:
            return None
        elif loop_vec.shape[0] < max_len:
            pad_vec = np.tile(EOS_VEC, max_len - loop_vec.shape[0]).reshape((-1, len(EOS_VEC)))
            loop_vec = np.concatenate([loop_vec, pad_vec], axis=0) # (max_len, 1 + N_ARGS)
        return loop_vec

    def draw(self, ax):
        colors = ['red', 'blue', 'green', 'brown', 'pink', 'yellow', 'purple', 'black'] * 10
        for i, curve in enumerate(self.children):
            curve.draw(ax, colors[i])

    def sample_points(self, n=32):
        points = np.stack([curve.sample_points(n) for curve in self.children], axis=0) # (n_curves, n, 2)
        return points

    def loop_distance(self,target_loop,scale:float):
        return point_distance(self.bbox*scale, target_loop.bbox*scale, type="l2")

    @staticmethod
    def match_primitives(gt_loop, pred_loop, scale: float, multiplier: int = 1):
        """
        Match primitives (e.g., curves) based on their bounding box distances.

        Args:
            gt_loop (object): Ground truth loop object.
            pred_loop (object): Predicted loop object.
            scale (float): The scaling factor.
            multiplier (int, optional): Multiplier for cost matrix. Defaults to 1.

        Returns:
            list: List containing matched ground truth and predicted curves.
        """
        if gt_loop is None:
            gt_curves=[None]
        else:
            gt_curves = gt_loop.children
        
        if pred_loop is None:
            pred_curves=[None]
        else:
            pred_curves = pred_loop.children

        n_gt = len(gt_curves)
        n_pred = len(pred_curves)
        n_max = max(n_gt, n_pred)

        # Initialize cost matrix with ones and apply multiplier
        cost_matrix = np.ones((n_max, n_max)) * multiplier

        # Pad lists with None if needed
        if n_gt < n_max:
            gt_curves += [None] * (n_max - n_gt)
        
        if n_pred < n_max:
            pred_curves += [None] * (n_max - n_pred)

        # Calculate Cost by calculating the distance between loops
        for ind_self in range(n_gt):
            for ind_pred in range(n_pred):
                if gt_curves[ind_self] is not None and pred_curves[ind_pred] is not None:
                    cost_matrix[ind_self, ind_pred] = gt_curves[ind_self].curve_distance(pred_curves[ind_pred], scale)

        # Use Hungarian matching to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # row_indices=list(row_indices)
        # col_indices=list(col_indices)
        # print(row_indices, col_indices)

        # Create a new pair with matched ground truth and predicted curves
        new_pair=create_matched_pair(list1=gt_curves,list2=pred_curves,
                                     row_indices=row_indices,col_indices=col_indices)
        return new_pair
    
class Profile(SketchBase):
    """Sketch profile，a closed region formed by one or more loops. 
    The outer-most loop is placed at first."""
    @staticmethod
    def from_dict(stat):
        all_loops = [Loop.from_dict(item) for item in stat['loops']]
        return Profile(all_loops)


    def __repr__(self):
        s = "Profile:"
        for loop in self.children:
            s += "\n    -" + str(loop)
        return s
    
    def __str__(self):
        return "Profile:" + "\n    -".join([str(loop) for loop in self.children])

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        all_loops = []
        command = vec[:, 0]
        end_idx = command.tolist().index(EOS_IDX)
        indices = np.where(command[:end_idx] == SOL_IDX)[0].tolist() + [end_idx]
        for i in range(len(indices) - 1):
            loop_vec = vec[indices[i]:indices[i + 1]]
            loop_vec = np.concatenate([loop_vec, EOS_VEC[np.newaxis]], axis=0)
            if loop_vec[0][0] == SOL_IDX and loop_vec[1][0] not in [SOL_IDX, EOS_IDX]:
                all_loops.append(Loop.from_vector(loop_vec, is_numerical=is_numerical))
        return Profile(all_loops)

    def reorder(self):
        if len(self.children) <= 1:
            return
        all_loops_bbox_min = np.stack([loop.bbox[0] for loop in self.children], axis=0).round(6)
        ind = np.lexsort(all_loops_bbox_min.transpose()[[1, 0]])
        self.children = [self.children[i] for i in ind]

    def draw(self, ax):
        for i, loop in enumerate(self.children):
            loop.draw(ax)
            ax.text(loop.start_point[0], loop.start_point[1], str(i))

    def to_vector(self, max_n_loops=None, max_len_loop=None, pad=True):
        loop_vecs = [loop.to_vector(None, add_eos=False) for loop in self.children]
        if max_n_loops is not None and len(loop_vecs) > max_n_loops:
            return None
        for vec in loop_vecs:
            if max_len_loop is not None and vec.shape[0] > max_len_loop:
                return None
        profile_vec = np.concatenate(loop_vecs, axis=0)
        profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis]], axis=0)
        if pad:
            pad_len = max_n_loops * max_len_loop - profile_vec.shape[0]
            profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return profile_vec

    def sample_points(self, n=32):
        points = np.concatenate([loop.sample_points(n) for loop in self.children], axis=0)
        return points

    @staticmethod
    def loop_match(gt_sketch, pred_sketch, scale: float,multiplier:int=2):
        """
        Match Loops according to the bounding box.

        Args:
            gt_sketch (object): The current object. (self must be ground truth)
            pred_sketch (object): The pred sketch object. (pred is prediction)
            scale (float): The scaling factor.
            multiplier (int): cost of distance with None

        Returns:
            list: List of matched loop pairs.
        """

        if pred_sketch is None:
            pred_loops=[None]
        else:
            pred_loops = copy.deepcopy(pred_sketch.children)

        if gt_sketch is None:
            gt_loops=[None]
        else:
            gt_loops = copy.deepcopy(gt_sketch.children)
        

        num_gt_loops = len(gt_loops)
        num_pred_loops = len(pred_loops)

        n_max = max(num_gt_loops, num_pred_loops)

        # Pad lists with None if needed
        if len(gt_loops) < n_max:
            gt_loops += [None] * (n_max - len(gt_loops))
        if len(pred_loops) < n_max:
            pred_loops += [None] * (n_max - len(pred_loops))

        cost_matrix = np.ones((n_max, n_max))*multiplier  # Fixed the shape of the cost matrix

        # Calculate Cost by calculating the distance between loops
        for ind_self in range(num_gt_loops):
            for ind_pred in range(num_pred_loops):
                if gt_loops[ind_self] is not None and pred_loops[ind_pred] is not None:
                    cost_matrix[ind_self, ind_pred] = gt_loops[ind_self].loop_distance(pred_loops[ind_pred], scale)

        # Use Hungarian matching to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Create matched loop pairs
        matched_loop_pair = create_matched_pair(list1=gt_loops, list2=pred_loops,
                                                row_indices=row_indices, col_indices=col_indices)


        # After loops are matched, match primitives 
        # (This will change the object from a pair of LoopSequences to a pair of list of CurveSequences)
        matched_curve_pair=[]
        for i, pair in enumerate(matched_loop_pair):
            matched_curve_pair+=Loop.match_primitives(pair[0],pair[1],scale,multiplier)
        
        return matched_curve_pair,matched_loop_pair