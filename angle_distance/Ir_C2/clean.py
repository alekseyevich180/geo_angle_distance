def clean_poscar(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    with open(output_filename, 'w') as f:
        # 写入文件头部信息
        f.write(lines[0])  # 材料名称
        f.write(lines[1])  # 晶胞比例
        f.write(lines[2])  # 晶胞向量第一行
        f.write(lines[3])  # 晶胞向量第二行
        f.write(lines[4])  # 晶胞向量第三行
        f.write(lines[5])  # 元素类型
        f.write(lines[6])  # 元素数量
        f.write("Direct\n")  # 强制写入 "Direct" 行

        # 处理原子坐标行
        start_line = 8
        for line in lines[start_line:]:
            # 跳过 'Direct' 关键字
            if line.strip() == "Direct":
                continue
            
            # 处理每一行，去掉每个原子后面的 'FFF' 和 'TTT'
            parts = line.strip().split()
            if len(parts) > 3:  # 确保有足够的部分（x, y, z 以及 'FFF'/'TTT'）
                cleaned_line = " ".join(parts[:3])  # 仅保留坐标部分，自动去除多余空格
                f.write(cleaned_line + "\n")

# 使用示例
input_file = 'angle_distance/Ir_C2/CONTCAR'  # 确保使用正斜杠
output_file = 'angle_distance/Ir_C2/CONTCAR_a'
clean_poscar(input_file, output_file)

print(f"Cleaned POSCAR saved to '{output_file}'")
