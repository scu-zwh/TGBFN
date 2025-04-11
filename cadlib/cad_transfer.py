import numpy as np 
import torch
import os  
if os.path.isdir('../cadlib/'): 
    os.chdir('../') 
    
from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD
from cadlib.macro import *
from cadlib.cad_dataset import normalize_data
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties

def save_steps(vec, path):
    bs, length = vec.shape
    for i in range(bs):
        item = vec[i]
        mat = vector_to_matrix(item)
        if mat is None:
            continue
        try:
            cad = CADSequence.from_vector(mat, is_numerical=True, n=256)
            cad3d = create_CAD(cad)
        except:
            continue
    
        save_solid_as_step(cad3d, path + f'/{i}.step')

def compute_surface_area_and_volume(shape):
    """
    计算三维形状的表面积和体积。

    :param shape: 要计算的三维形状 (TopoDS_Shape 对象)
    :return: (表面积, 体积)
    """
    
    # 创建 GProp_GProps 对象
    surface_props = GProp_GProps()
    volume_props = GProp_GProps()

    # 计算表面积
    brepgprop_SurfaceProperties(shape, surface_props)
    surface_area = surface_props.Mass()  # 表面积存储在 Mass() 方法中

    # 计算体积
    brepgprop_VolumeProperties(shape, volume_props, True, False)
    volume = volume_props.Mass()  # 体积存储在 Mass() 方法中

    return surface_area, volume

def save_solid_as_step(solid, filename):
    if solid.IsNull():
        print("solid 对象无效！")
        return
    
    step_writer = STEPControl_Writer()
    step_writer.Transfer(solid, STEPControl_AsIs)
    status = step_writer.Write(filename)
    
    if status == 1:  # 1 表示成功
        print("STEP 文件保存成功：", filename)
    else:
        print("STEP 文件保存失败！")

def get_row(vector, i, row_length):
    if vector[i] == SOL_IDX:
        current_row = [vector[i]]
        # 将当前行填充至 17 个元素，并添加到行列表
        current_row += [-1] * (row_length - len(current_row))
        i += 1
        return i, current_row
    
    elif vector[i] == LINE_IDX:
        if i + 2 >= len(vector):  # 确保有足够的元素
            return None, None
        current_row = [vector[i], vector[i+1], vector[i+2]]
        i += 3
        current_row += [-1] * (row_length - len(current_row))
        return i, current_row
    
    elif vector[i] == ARC_IDX:
        if i + 4 >= len(vector):  # 确保有足够的元素
            return None, None
        current_row = [vector[i], vector[i+1], vector[i+2], vector[i+3], vector[i+4]]
        i += 5
        current_row += [-1] * (row_length - len(current_row))
        return i, current_row
    
    elif vector[i] == CIRCLE_IDX:
        if i + 3 >= len(vector):  # 确保有足够的元素
            return None, None
        current_row = [vector[i], vector[i+1], vector[i+2], -1, -1, vector[i+3]]
        i += 4
        current_row += [-1] * (row_length - len(current_row))
        return i, current_row   
             
    elif vector[i] == EXT_IDX:
        if i + 11 >= len(vector):  # 确保有足够的元素
            return None, None
        current_row = [vector[i], -1, -1, -1, -1, -1, vector[i+1], vector[i+2], vector[i+3], vector[i+4], vector[i+5], vector[i+6],
                        vector[i+7], vector[i+8], vector[i+9], vector[i+10], vector[i+11]]
        i += 12
        return i, current_row   
    
    elif vector[i] == EOS_IDX:
        current_row = [vector[i]]
        current_row += [-1] * (row_length - len(current_row))
        i += 1
        return i, current_row           

def vector_to_matrix(vector, row_length=17):
    vector = vector[vector != PAD_IDX]
    if vector[0] != SOL_IDX or vector[-1] != EOS_IDX:
        return None
        
    rows = []
    
    i = 0
    while i < len(vector):
        if vector[i] not in [LINE_IDX, ARC_IDX, CIRCLE_IDX, EOS_IDX, SOL_IDX, EXT_IDX]:
            return None
        i, current_row = get_row(vector, i, row_length)
        if i is None:
            return None
        rows.append(current_row)        
    
    # 将结果转换为 numpy 数组
    return np.array(rows)

def vec2sv(vec, is_mat=False):
    if is_mat:
        mat = vec
    else:
        mat = vector_to_matrix(vec)
    if mat is None:
        return -1, -1
    cad = CADSequence.from_vector(mat, is_numerical=True, n=256)
    cad3d = create_CAD(cad)
    
    return compute_surface_area_and_volume(cad3d)

def matrix_to_vector(matrix):
    # 将矩阵展开为一维数组
    flattened_array = matrix.flatten()
    # 过滤掉值为 -1 的元素
    result_vector = flattened_array[flattened_array != -1]
    return result_vector

def get_output_sv(samples):
    output_sv = []
    mat = samples.cpu().to(torch.int32).numpy()
    valid_count = 0
    for output_vec in mat:
        # print(output_vec)
        try:
            area, vol = vec2sv(output_vec, is_mat=False)
        except:
            area, vol = -1, -1
        if not area == vol == -1:
            valid_count += 1
            sv_data = {"area": area, "vol": vol}
            sv_data = normalize_data(sv_data)
            area, vol = sv_data["area"], sv_data["vol"]
        output_sv.append([area, vol])  
    return output_sv, valid_count

def vec2step(vec, is_mat=False, file_path=""):
    if is_mat:
        mat = vec
    else:
        mat = vector_to_matrix(vec)
    if mat is None:
        return -1, -1
    cad = CADSequence.from_vector(mat, is_numerical=True, n=256)
    cad3d = create_CAD(cad)
    
    save_solid_as_step(cad3d, file_path)
    
        
