import os
import numpy as np
import random
from scipy.spatial import cKDTree
import sys
sys.path.append("..")
from utils import read_ply
from .visualize import vec2CADsolid, CADsolid2pc
from .instructions_to_arr import convert_string_to_array

def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = cKDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = cKDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points

def normalize_pc_text2cad(points):
    scale = np.abs(np.max(points) - np.min(points))
    points = points / scale
    return points


def process_single_vec_pc(out_vec, gt_pc, n_points=2000):
    """输入预测的vec和提前准备好的真实的pc"""
    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        # print(str(e))
        print("create_CAD failed")
        return None

    try:
        out_pc = CADsolid2pc(shape, n_points)
    except Exception as e:
        # print(str(e))
        print("convert pc failed")
        return None

    # if np.max(np.abs(out_pc)) > 2:  # normalize out-of-bound data
    #     out_pc = normalize_pc(out_pc)
    gt_pc = normalize_pc_text2cad(gt_pc)
    out_pc = normalize_pc_text2cad(out_pc)

    sample_idx = random.sample(list(range(gt_pc.shape[0])), n_points)
    gt_pc = gt_pc[sample_idx]

    cd = chamfer_dist(gt_pc, out_pc)
    return cd

def process_single_vec_vec(out_vec, gt_vec, n_points=2000):
    """输入预测的和真实的vec"""
    try:
        out_shape = vec2CADsolid(out_vec)
        gt_shape = vec2CADsolid(gt_vec)
    except Exception as e:
        # print(str(e))
        print("create_CAD failed")
        return None

    try:
        out_pc = CADsolid2pc(out_shape, n_points)
        gt_pc = CADsolid2pc(gt_shape, n_points)
    except Exception as e:
        # print(str(e))
        print("convert pc failed")
        return None

    # if np.max(np.abs(out_pc)) > 2:  # normalize out-of-bound data
    #     out_pc = normalize_pc(out_pc)
    gt_pc = normalize_pc_text2cad(gt_pc)
    out_pc = normalize_pc_text2cad(out_pc)

    sample_idx = random.sample(list(range(gt_pc.shape[0])), n_points)
    gt_pc = gt_pc[sample_idx]

    cd = chamfer_dist(gt_pc, out_pc)
    return cd

from .cad_transfer import vector_to_matrix

def compute_cd(gt_vec, pred_vec):
    dists = []
    invalid_count = 0
    for gt_item, pred_item in zip(gt_vec, pred_vec):
        try:
            gt_mat, pred_mat = vector_to_matrix(gt_item), vector_to_matrix(pred_item)
        # print(gt_text)
            # pred_vec = convert_string_to_array(pred_text)
            # gt_vec = convert_string_to_array(gt_text)
            if gt_vec.shape == (0,):
                continue
            # print('process_single_vec_vec')
            cd = process_single_vec_vec(pred_mat, gt_mat, n_points=2000)
            dists.append(cd)
        except Exception as e: # 不合法的输出
            # print(str(e))
            invalid_count += 1
            pass
    
    valid_dists = [x for x in dists if x is not None]
    invalid_count += (len(dists) - len(valid_dists))
    invalid_ratio = invalid_count / gt_vec.shape[0]
    
    avg_dist = np.mean(valid_dists)
    med_dist = np.median(valid_dists)
 

    print(f'total: {gt_vec.shape[0]}, invalid_count: {invalid_count}, invalid_ratio: {invalid_ratio}')
    return avg_dist, med_dist, invalid_ratio