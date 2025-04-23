import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
dgl = pd.read_csv("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/dgl.csv")
pyg = pd.read_csv("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/pyg.csv")
mgat16 = pd.read_csv("/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/mgat16.csv")

# 定义列名
s_cols = ['s0', 's1', 's2']
a_cols = ['a0', 'a1', 'a2']

# 重构计算逻辑
def compute_speedup(base_df, target_df, base_label):
    records = []
    for col_group, stage in zip([s_cols, a_cols], ['SpMM', 'Attention']):
        for col in col_group:
            for i in range(len(base_df)):
                base = base_df[col][i]
                target = target_df[col][i]
                if target != 0:
                    speedup = base / target
                    records.append({
                        'Against': base_label,
                        'Stage': stage,
                        'Speedup': speedup
                    })
    return records

# 合并数据
records = []
records += compute_speedup(dgl, mgat16, 'DGL')
records += compute_speedup(pyg, mgat16, 'PyG')
df_plot = pd.DataFrame(records)

# 创建联合列用于分类绘图：'DGL-SpMM'、'DGL-Attention' 等
df_plot['Group'] = df_plot['Against'] + '-' + df_plot['Stage']

# 画图
plt.figure(figsize=(10, 4))
sns.violinplot(data=df_plot, x='Group', y='Speedup', inner='point', palette='Set2')
plt.title("FP16 Speedup over DGL and PyG (Separated by Stage)")
plt.ylabel("Speedup")
plt.ylim(0.5, 2.5)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('/home/shijinliang/module/tpds/ATT/end2end_break/gat_final/result/gat_spmm.png', dpi=800)
