import numpy as np
import matplotlib.pyplot as plt

# 数据
bars = [1, 2, 4, 8]
values1 = [50, 70, 100, 150]
values2 = [60, 80, 120, 180]
values3 = [70, 90, 130, 200]

# 计算位置
bar_width = 0.2
x = np.arange(len(bars))

# 创建图形
fig, ax = plt.subplots(figsize=(7, 5))

# 绘制柱状图
rects1 = ax.bar(x - bar_width, values1, bar_width, label='Group 1', color='lightgreen', edgecolor='black')
rects2 = ax.bar(x, values2, bar_width, label='Group 2', color='khaki', edgecolor='black')
rects3 = ax.bar(x + bar_width, values3, bar_width, label='Group 3', color='lightblue', edgecolor='black', hatch='//')

# 添加标注
for i, v in enumerate(values1):
    ax.text(x[i] - bar_width, v + 5, f'1x', ha='center', fontsize=10)

for i, v in enumerate(values2):
    ax.text(x[i], v + 5, f'1.2x', ha='center', fontsize=10)

for i, v in enumerate(values3):
    ax.text(x[i] + bar_width, v + 5, f'1.39x', ha='center', fontsize=10)

# 添加横纵坐标标签和标题
ax.set_xlabel('Number of GPUs')
ax.set_ylabel('TFLOPS / GPU')
ax.set_title('Performance per GPU')

# 设置 x 轴刻度
ax.set_xticks(x)
ax.set_xticklabels(bars)

# 显示图例
ax.legend()

# 显示网格
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

ax.xaxis.set_visible(False)
plt.savefig('/home/shijinliang/module/tpds/ATT/end2end/plot/end2end/test.png', dpi=800)
