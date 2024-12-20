import numpy as np

def calculate_plane_normal(point1, point2, point3, scale_factor=1.0):
    """
    根据三点定义平面，计算平面法向量。
    
    参数:
        - point1, point2, point3: numpy 数组，表示平面上的三点坐标。
        - scale_factor: float，用于调整点之间的权重，默认值为 1.0。
    
    返回:
        - normal: numpy 数组，归一化的平面法向量。
    """
    # 按照 scale_factor 调整点的权重
    vector1 = point2 - scale_factor * point1
    vector2 = point3 - point1

    # 计算法向量并归一化
    normal = np.cross(vector1, vector2)
    normal = normal / np.linalg.norm(normal)  # 归一化
    return normal
