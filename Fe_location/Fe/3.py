import os
import glob

folder_path = "C:\\Users\\wu\\Desktop\\python_geo\\Fe_location\\Fe"
file_pattern = os.path.join(folder_path, "CONTCAR_*.vasp")
files = glob.glob(file_pattern)

print(f"当前路径: {folder_path}")
print(f"匹配模式: {file_pattern}")
print(f"找到的文件: {files}")
