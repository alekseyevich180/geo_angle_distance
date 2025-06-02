import matplotlib.pyplot as plt

# 设定原子位置（虽然不画，但保持一致性）
atoms = {
    'O': {'pos': (0.0, 0.0), 'color': 'orange'},
    'H1': {'pos': (-0.76, 0.58), 'color': 'skyblue'},
    'H2': {'pos': (0.76, 0.58), 'color': 'skyblue'}
}

# 创建画布，比例仍保持2:1
fig, ax = plt.subplots(figsize=(14, 7))  # 稍微收窄宽度以适配调整后的表格

# 空白区域设置
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 2)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Water Molecule (H₂O) and Coulomb Matrix (Symbolic Form)', fontsize=20, pad=20)

# Coulomb Matrix内容（公式表达式）
symbolic_matrix = [
    ["0.5 $Z_O^{2.4}$", "$\\frac{Z_O Z_H}{r_{OH1}}$", "$\\frac{Z_O Z_H}{r_{OH2}}$"],
    ["$\\frac{Z_O Z_H}{r_{OH1}}$", "0.5 $Z_H^{2.4}$", "$\\frac{Z_H Z_H}{r_{H1H2}}$"],
    ["$\\frac{Z_O Z_H}{r_{OH2}}$", "$\\frac{Z_H Z_H}{r_{H1H2}}$", "0.5 $Z_H^{2.4}$"]
]
col_labels = row_labels = ['O', 'H1', 'H2']

# 绘制表格，缩窄单元格宽度
table = plt.table(cellText=symbolic_matrix,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  colWidths=[0.15]*3,  # 缩小单元格宽度
                  loc='center',
                  cellLoc='center')
table.scale(3.5, 4.5)  # 控制表格整体大小
table.auto_set_font_size(False)
table.set_fontsize(18)  # 保持较大字体，清晰可读

plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)  # 调整边距
plt.show()
