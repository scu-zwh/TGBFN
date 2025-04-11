import numpy as np
from .macro import *
from .sketch import Profile
from .curves import Line, Arc, Circle
from .instructions_to_arr import convert_string_to_array
import torch
from .cad_transfer import vector_to_matrix


def expand_cad(cad_vec, max_length=64):
    cad_vec = cad_vec[cad_vec != 262]
    
    if cad_vec[0] != 260:
        bos = torch.tensor([260])
        cad_vec = torch.concat([bos, cad_vec])
    length = cad_vec.shape[0]
    if length >= max_length:
        length = max_length-1
    # result = torch.tensor([], dtype=cad_vec.dtype)
    for i in range(length):
        if cad_vec[i] == 256:
            if i+2 >= length:
                break
            temp = torch.tensor([0, cad_vec[i+1], cad_vec[i+2], -1, -1, -1, -1, -
                                1, -1, -1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)
            result = torch.concat([result, temp], dim=0)
            i += 2
        elif cad_vec[i] == 257:
            if i+4 >= length:
                break
            temp = torch.tensor([1, cad_vec[i+1], cad_vec[i+2], cad_vec[i+3], cad_vec[i+4], -
                                1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)
            result = torch.concat([result, temp], dim=0)
            i += 4
        elif cad_vec[i] == 258:
            if i+3 >= length:
                break
            temp = torch.tensor([2, cad_vec[i+1], cad_vec[i+2], -1, -1, cad_vec[i+3], -
                                1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)
            result = torch.concat([result, temp], dim=0)
            i += 3
        elif cad_vec[i] == 259:
            temp = torch.tensor([3, -1, -1, -1, -1, -1, -1, -1, -1, -
                                1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)
            result = torch.concat([result, temp], dim=0)
            break
        elif cad_vec[i] == 260:
            temp = torch.tensor([4, -1, -1, -1, -1, -1, -1, -1, -1, -
                                1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)
            result = temp
        elif cad_vec[i] == 261:
            if i+11 >= length:
                break
            temp = torch.tensor([5, -1, -1, -1, -1, -1, cad_vec[i+1], cad_vec[i+2], cad_vec[i+3], cad_vec[i+4], cad_vec[i+5],
                                cad_vec[i+6], cad_vec[i+7], cad_vec[i+8], cad_vec[i+9], cad_vec[i+10], cad_vec[i+11]]).unsqueeze(0)
            result = torch.concat([result, temp], dim=0)
            i += 11
        elif cad_vec[i] == 262:
            temp = torch.tensor([6, -1, -1, -1, -1, -1, -1, -1, -1, -
                                1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)
            result = torch.concat([result, temp], dim=0)
    # temp = torch.tensor([3, -1, -1, -1, -1, -1, -1, -1, -1, -
    #                     1, -1, -1, -1, -1, -1, -1, -1], dtype=cad_vec.dtype).cuda()
    # result = torch.concat([result, temp], dim=1)
    return np.array(result)


def extract_profiles(arr: np.ndarray):
    """
    从形状为 (m, 17) 的 numpy 数组中，提取所有的 profile 并返回。

    规则：
    1) profile 的开始：遇到 SOL (command_type == 4)，该行加入 profile。
    2) profile 的结束：遇到 Ext (command_type == 5) 或 EOS (command_type == 3)，
       不把当前这行 (Ext/EOS) 放进 profile；然后检查 profile 末行有没有 EOS，
       若无则补上一行 EOS_VEC。将此 profile 存入结果列表。
       如果是 EOS，则整个序列结束（题目保证只有一个 EOS，且在末行）。
    3) 一个 profile 内部可能含有多个 SOL 行，但都属于同一个 profile，直到遇到 Ext 或 EOS 才结束。
    4) 返回值： (profiles_list, profiles_count)

    参数
    ----
    arr : np.ndarray
        m×17 的原始指令数组。

    返回
    ----
    profiles : list of np.ndarray
        分割得到的所有 profile，每个 profile 是一个 p×17 的子数组（p 的大小随内容而变）。
    n_profiles : int
        profile 的数量。

    示例
    ----
    >>> # arr 的第一列代表命令类型，假设有如下伪数据：
    >>> arr = np.array([
    ...     [4, ...],  # SOL
    ...     [0, ...],  # Line
    ...     [4, ...],  # SOL (同一个 profile 内部再次 start of loop)
    ...     [1, ...],  # Arc
    ...     [5, ...],  # Ext，结束当前 profile
    ...     [4, ...],  # 新的 SOL => 下一个 profile
    ...     [2, ...],  # Circle
    ...     [3, ...],  # EOS, 整个序列结束
    ... ])
    >>> profiles, n_profiles = extract_profiles(arr)
    >>> for i, pf in enumerate(profiles):
    ...     print(f"Profile {i} shape:", pf.shape)
    ...     print(pf)
    """
    profiles = []
    current_profile = []  # 用于收集当前 profile 的所有行
    in_profile = False    # 标记是否正在收集 profile

    for row in arr:
        cmd_type = row[0]

        if cmd_type ==SOL_IDX:  # SOL
            # 如果当前还没开始收集 profile，则开启新的 profile
            if not in_profile:
                in_profile = True
                current_profile = [row]
            else:
                # 如果已经在 profile 中，则直接加入
                current_profile.append(row)

        elif cmd_type in [LINE_IDX, ARC_IDX, CIRCLE_IDX]:  # Line, Arc, Circle 等
            # 如果在 profile 中，直接加入
            if in_profile:
                current_profile.append(row)

        elif cmd_type == EXT_IDX:  # Ext
            # 结束当前 profile（若在收集中）
            if in_profile:
                # 若最后一行不是 EOS，补一个
                if len(current_profile) == 0 or current_profile[-1][0] != EOS_IDX:
                    current_profile.append(EOS_VEC.copy())
                # 存入结果
                profiles.append(np.vstack(current_profile))
                # 重置
                current_profile = []
                in_profile = False
            # Ext 行本身不加入任何 profile

        elif cmd_type == EOS_IDX:  # EOS
            # 结束当前 profile（若在收集中）
            if in_profile:
                if len(current_profile) == 0 or current_profile[-1][0] != EOS_IDX:
                    current_profile.append(EOS_VEC.copy())
                profiles.append(np.vstack(current_profile))
                current_profile = []
                in_profile = False
            # EOS 行本身不加入 profile，并且整个序列结束
            break

        else:
            # 万一出现未知的 command_type，按需求可选择如何处理，这里暂时忽略
            pass

    # 万一原始数据中，没有遇到 EOS 就结束了（题意中说最后必有 EOS，这里做个保护）
    if in_profile and len(current_profile) > 0:
        # 补 EOS
        if current_profile[-1][0] != EOS_IDX:
            current_profile.append(EOS_VEC.copy())
        profiles.append(np.vstack(current_profile))

    return profiles, len(profiles)

def match_profiles_from_arrays(gt_vec: np.ndarray, pred_vec: np.ndarray, scale: float=1.):
    """
    给定两个 m×17 的 numpy 数组:
      1) 分别调用 extract_profiles 得到各自的 profiles (list of p×17 的数组)。
      2) 逐一配对比较(第i个与第i个)，若一方越界，则用 EOS_VEC_2D 代替。
      3) 对每一次比较，调用 Profile.loop_match() 得到 matched_curve_pair, matched_loop_pair。
      4) 收集所有 matched_curve_pair 到一个列表 all_matched_curve_pairs 中。
      5) 返回 all_matched_curve_pairs (及所有 matched_loop_pair，如需要可一并返回)。

    Parameters
    ----------
    gt_vec : np.ndarray
        Ground truth 的 m×17 矩阵
    pred_vec : np.ndarray
        Prediction 的 m×17 矩阵
    scale: float
        缩放尺度

    Returns
    -------
    all_matched_curve_pairs : list
        收集所有 matched_curve_pair 的列表
    
    Notes
    -----
    你可以根据项目需要，扩展返回值，或对 EOS_VEC_2D 的形式做自定义。
    """
    EOS_VEC_2D = np.expand_dims(EOS_VEC, axis=0)  # 在第0轴增加一个维度

    # 1) 分别提取 profiles
    gt_profiles, gt_count = extract_profiles(gt_vec)
    pred_profiles, pred_count = extract_profiles(pred_vec)
    # 建立一个最终的列表
    all_matched_curve_pairs = []
    all_matched_loop_pairs = []  # 若你也想收集 loop-pair，可以保留这一并返回

    # 2) 遍历数量为二者的最大值
    max_count = max(gt_count, pred_count)
    for i in range(max_count):
        # 取第 i 个 ground truth 的 profile
        if i < gt_count:
            gt_profile_vec = gt_profiles[i]
        else:
            # 若 gt 侧已经没有更多 profile，就用 EOS_VEC_2D 代替
            gt_profile_vec = EOS_VEC_2D

        # 取第 i 个 prediction 的 profile
        if i < pred_count:
            pred_profile_vec = pred_profiles[i]
        else:
            # 若 pred 侧已经没有更多 profile，就用 EOS_VEC_2D 代替
            pred_profile_vec = EOS_VEC_2D

        # 3) 转成 Profile 对象
        gt_profile = Profile.from_vector(gt_profile_vec)
        pred_profile = Profile.from_vector(pred_profile_vec)

        # 4) 调用 loop_match
        matched_curve_pair, matched_loop_pair = Profile.loop_match(
            gt_profile, 
            pred_profile, 
            scale=scale
        )

        # 5) 收集到列表中
        #    注意 matched_curve_pair 通常是一个 list，可以直接 extend
        all_matched_curve_pairs.extend(matched_curve_pair)
        # 如果你也想收集 matched_loop_pair，可以这样
        all_matched_loop_pairs.extend(matched_loop_pair)

    # 你可以只返回 curve_pairs，也可以把 loop_pairs 也返回
    return all_matched_curve_pairs, all_matched_loop_pairs

def compute_curve_f1_single(all_curve_pairs):
    """
    给定一个 all_curve_pairs 列表（里面的每个元素都是 [gt_obj, pred_obj]），
    其中 gt_obj, pred_obj 可能是 Line, Arc, Circle 或者 None。
    计算并返回三种操作(Line, Arc, Circle)的 F1 分数。
    """
    # 统计量：分别统计每个类型在左侧/右侧出现的次数，以及左右同为该类型的次数
    line_left = line_right = line_both = 0
    arc_left = arc_right = arc_both = 0
    circle_left = circle_right = circle_both = 0

    for pair in all_curve_pairs:
        gt_obj, pred_obj = pair[0], pair[1]

        # ---------- 处理 Line ----------
        if isinstance(gt_obj, Line):
            line_left += 1
        if isinstance(pred_obj, Line):
            line_right += 1
        if isinstance(gt_obj, Line) and isinstance(pred_obj, Line):
            line_both += 1

        # ---------- 处理 Arc ----------
        if isinstance(gt_obj, Arc):
            arc_left += 1
        if isinstance(pred_obj, Arc):
            arc_right += 1
        if isinstance(gt_obj, Arc) and isinstance(pred_obj, Arc):
            arc_both += 1

        # ---------- 处理 Circle ----------
        if isinstance(gt_obj, Circle):
            circle_left += 1
        if isinstance(pred_obj, Circle):
            circle_right += 1
        if isinstance(gt_obj, Circle) and isinstance(pred_obj, Circle):
            circle_both += 1

    def safe_div(num, den):
        return num / den if den != 0 else 0

    def compute_f1(num_both, num_left, num_right):
        """
        num_both = 左右都为某类型的数量
        num_left = 左侧为该类型的数量
        num_right= 右侧为该类型的数量
        """
        precision = safe_div(num_both, num_right)  # 右侧预测多少该类型中，实际也为该类型的比例
        recall = safe_div(num_both, num_left)      # 左侧实际为该类型中，被预测对的比例
        denom = (precision + recall)
        if denom == 0:
            return 0.0
        return 2 * precision * recall / denom

    # 分别计算三个操作的 F1
    line_f1 = compute_f1(line_both, line_left, line_right)
    arc_f1 = compute_f1(arc_both, arc_left, arc_right)
    circle_f1 = compute_f1(circle_both, circle_left, circle_right)

    return line_f1, arc_f1, circle_f1

def compute_ext_f1_single(pred_vec, gt_vec):
    # 计算 num_ext_gt 和 num_ext_pred
    num_ext_gt = np.sum(gt_vec[:, 0] == EXT_IDX)
    num_ext_pred = np.sum(pred_vec[:, 0] == EXT_IDX)
    
    # 计算 num_ext = min(num_ext_gt, num_ext_pred)
    num_ext = min(num_ext_gt, num_ext_pred)
    
    # 计算 ext_recall 和 ext_precision
    if num_ext_gt == 0:
        ext_recall = 0
    else:
        ext_recall = num_ext / num_ext_gt
    
    if num_ext_pred == 0:
        ext_precision = 0
    else:
        ext_precision = num_ext / num_ext_pred
    
    # 计算 F1 值
    if ext_recall + ext_precision == 0:
        ext_f1 = 0.
    else:
        ext_f1 = 2 * ext_recall * ext_precision / (ext_recall + ext_precision)
    # print(ext_precision, ext_recall, ext_f1)
    return ext_f1

def compute_f1_all(gt_vec, pred_vec):
    all_line_f1, all_arc_f1, all_circle_f1 = [], [], []
    all_extrude_f1 = []
    invalid_count = 0
    for gt_item, pred_item in zip(gt_vec, pred_vec):
        try:
            gt_mat, pred_mat = vector_to_matrix(gt_item), vector_to_matrix(pred_item)
            # print(pred_vec.shape, pred_vec)
            if gt_mat.shape == (0,): # Skip this!
                continue
            assert pred_mat.shape[1] == 17
        except Exception as e: # 不合法的输出
            invalid_count += 1
            pass
        else:
            curve_pairs, _ = match_profiles_from_arrays(gt_vec=gt_mat, pred_vec=pred_mat)
            line_f1, arc_f1, circle_f1 = compute_curve_f1_single(curve_pairs)
            extrude_f1 = compute_ext_f1_single(pred_mat, gt_mat)
            all_line_f1.append(line_f1)
            all_arc_f1.append(arc_f1)
            all_circle_f1.append(circle_f1)
            all_extrude_f1.append(extrude_f1)

    # 将列表转换为 NumPy 数组
    line_f1_array = np.array(all_line_f1)
    arc_f1_array = np.array(all_arc_f1)
    circle_f1_array = np.array(all_circle_f1)
    extrude_f1_array = np.array(all_extrude_f1)

    # 对于Curve：过滤掉 F1 分数为 0 的项，并计算非零 F1 的平均值
    filtered_line_f1 = line_f1_array[line_f1_array > 0]
    filtered_arc_f1 = arc_f1_array[arc_f1_array > 0]
    filtered_circle_f1 = circle_f1_array[circle_f1_array > 0]

    avg_line_f1 = filtered_line_f1.mean() if filtered_line_f1.size > 0 else 0
    avg_arc_f1 = filtered_arc_f1.mean() if filtered_arc_f1.size > 0 else 0
    avg_circle_f1 = filtered_circle_f1.mean() if filtered_circle_f1.size > 0 else 0

    # 对于Extrude：直接平均
    avg_extrude_f1 = extrude_f1_array.mean()
    return avg_line_f1, avg_arc_f1, avg_circle_f1, avg_extrude_f1
    