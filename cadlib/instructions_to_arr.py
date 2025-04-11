import numpy as np

def instructions_to_array(instructions):
    instr_map = ["Line", "Arc", "Circle", "EOS", "SOL", "Extrude"]
    b_map = ["NewBody", "Join", "Cut", "Intersect"]
    u_map = ["OneSide", "Symmetric", "TwoSides"]

    rows = []

    for line in instructions:
        line = line.strip()
        # 初始化一行数据全为-1
        row_data = [-1]*17
        
        if line.startswith("<"):
            # 尖括号指令（EOS, SOL）
            end_bracket = line.find('>')
            instr_name = line[1:end_bracket]  # 拿到SOL或EOS
            instr_type_idx = instr_map.index(instr_name)
            row_data[0] = instr_type_idx
        else:
            # 非尖括号指令 (Line, Arc, Circle, Extrude)
            if ':' in line:
                name_part, param_part = line.split(':', 1)
                name_part = name_part.strip()
                param_part = param_part.strip()
            else:
                # 没有冒号则无参数
                name_part = line
                param_part = ""
            
            if '_' in name_part:
                base_name = name_part.split('_',1)[0]
            else:
                base_name = name_part
            
            if base_name not in instr_map:
                raise ValueError(f"未知的指令类型: {base_name}")
            instr_type_idx = instr_map.index(base_name)
            row_data[0] = instr_type_idx
            
            # 解析参数
            params = []
            if '(' in param_part and ')' in param_part:
                start_paren = param_part.find('(')
                end_paren = param_part.find(')')
                param_str = param_part[start_paren+1:end_paren].strip()
                if param_str:
                    # 这里不能简单split(',')后strip，要小心空格
                    # 一般用split(',')再strip
                    params = [p.strip() for p in param_str.split(',')]

            instr_type = instr_map[instr_type_idx]
            if instr_type == "Line":
                # (x, y)
                if len(params) > 0:
                    row_data[1] = int(float(params[0]))
                if len(params) > 1:
                    row_data[2] = int(float(params[1]))

            elif instr_type == "Arc":
                # (x,y,alpha,f)
                if len(params) > 0:
                    row_data[1] = int(float(params[0]))
                if len(params) > 1:
                    row_data[2] = int(float(params[1]))
                if len(params) > 2:
                    row_data[3] = int(float(params[2]))
                if len(params) > 3:
                    row_data[4] = int(float(params[3]))

            elif instr_type == "Circle":
                # (x,y,r)
                if len(params) > 0:
                    row_data[1] = int(float(params[0]))
                if len(params) > 1:
                    row_data[2] = int(float(params[1]))
                if len(params) > 2:
                    row_data[5] = int(float(params[2]))

            elif instr_type == "EOS":
                # 无参数
                pass

            elif instr_type == "SOL":
                # 无参数
                pass

            elif instr_type == "Extrude":
                # (θ, φ, γ, p_x, p_y, p_z, s, e_1, e_2, b, u)
                # 前9个数字，后2个字符串映射回索引
                def safe_int(p):
                    return int(p) if p.isdigit() or (p.startswith('-') and p[1:].isdigit()) else -1
                
                # θ(6), φ(7), γ(8), p_x(9), p_y(10), p_z(11), s(12), e_1(13), e_2(14)
                if len(params) > 0:
                    row_data[6] = safe_int(params[0])
                if len(params) > 1:
                    row_data[7] = safe_int(params[1])
                if len(params) > 2:
                    row_data[8] = safe_int(params[2])
                if len(params) > 3:
                    row_data[9] = safe_int(params[3])
                if len(params) > 4:
                    row_data[10] = safe_int(params[4])
                if len(params) > 5:
                    row_data[11] = safe_int(params[5])
                if len(params) > 6:
                    row_data[12] = safe_int(params[6])
                if len(params) > 7:
                    row_data[13] = safe_int(params[7])
                if len(params) > 8:
                    row_data[14] = safe_int(params[8])

                # b参数
                if len(params) > 9:
                    b_str = params[9]
                    if b_str in b_map:
                        row_data[15] = b_map.index(b_str)
                    else:
                        row_data[15] = -1
                # u参数
                if len(params) > 10:
                    u_str = params[10]
                    if u_str in u_map:
                        row_data[16] = u_map.index(u_str)
                    else:
                        row_data[16] = -1

        rows.append(row_data)

    return np.array(rows, dtype=int)


def convert_string_to_array(instr_str):
    # 将输入的多行字符串转为指令列表
    lines = instr_str.splitlines()
    # 去掉前后空格，并过滤掉空行
    instructions = [line.strip() for line in lines if line.strip() != '']
    # 调用instructions_to_array处理
    return instructions_to_array(instructions)


# 示例
if __name__ == "__main__":
    test_str = """<SOL>_1\n    Line_2: (128, 134)\n    Arc_3: (128, 217, 128, 0)\n    Line_4: (128, 223)\n    Arc_5: (128, 128, 128, 1)\n    Extrude_6: (128, 128, 128, 128, 117, 128, 22, 180, 128, NewBody, OneSide)\n<EOS>_7
"""
    arr = convert_string_to_array(test_str)
    print(arr)

