import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = '1c3s5z.csv'
data = pd.read_csv(file_path)

groups = {
 'LF-Commer': [
'commformer_1c3s5z_single_lf_seed1 - incre_win_rate',
 'commformer_1c3s5z_single_lf_seed2 - incre_win_rate',
 'commformer_1c3s5z_single_lf_seed3 - incre_win_rate',
 ],
 'MAPPO': [
 'mappo_1c3s5z_10v10_seed1 - incre_win_rate',
'mappo_1c3s5z_10v10_seed2 - incre_win_rate',
 'mappo_1c3s5z_10v10_seed3 - incre_win_rate',
],
 'HAPPO': [
 'happo_1c3s5z_10v10_seed1 - incre_win_rate',
'happo_1c3s5z_10v10_seed2 - incre_win_rate',
 'happo_1c3s5z_10v10_seed3 - incre_win_rate',
]
}

df_list = []
for algo_name, seed_cols in groups.items():
 for seed_col in seed_cols:
    temp_df = data[['Step', seed_col]].copy()
    temp_df.rename(columns={seed_col: 'Win_Rate'}, inplace=True)
    temp_df['Algorithm'] = algo_name
    df_list.append(temp_df)

df_long = pd.concat(df_list, ignore_index=True)


df_long['Step'] = pd.to_numeric(df_long['Step'], errors='coerce')
df_long['Win_Rate'] = pd.to_numeric(df_long['Win_Rate'], errors='coerce')
df_long.dropna(inplace=True)

df_stats = (
    df_long.groupby(['Step', 'Algorithm'], as_index=False).agg( mean_winrate=('Win_Rate', 'mean'), std_winrate=('Win_Rate', 'std') )
)


df_stats['lower'] = (df_stats['mean_winrate'] - df_stats['std_winrate']).clip(0, 1)
df_stats['upper'] = (df_stats['mean_winrate'] + df_stats['std_winrate']).clip(0, 1)
df_stats['mean_clipped'] = df_stats['mean_winrate'].clip(0, 1)

sns.set_theme(
 style="whitegrid",
 context="talk",
 rc={
 "figure.figsize": (10, 6),
 "figure.dpi": 100,
 "grid.color": "#DDDDDD", # 网格颜色
 "grid.alpha": 0.5 # 网格透明度
 }
)

fig, ax = plt.subplots()

# 颜色自定义
palette = {
    'LF-Commer': '#a0d3ff',
    'MAPPO': '#ffb3de', 
    'HAPPO': '#b39ddb',
}

# 绘制均值曲线和置信区间
for algo in df_stats['Algorithm'].unique():
    sub = df_stats[df_stats['Algorithm'] == algo]
    # 转换为numpy数组以避免兼容性问题
    x = sub['Step'].to_numpy()
    y = sub['mean_clipped'].to_numpy()
    lower = sub['lower'].to_numpy()
    upper = sub['upper'].to_numpy()
    
    ax.plot(x, y, label=algo, color=palette.get(algo, None))
    ax.fill_between(x, lower, upper, alpha=0.2, color=palette.get(algo, None))

ax.set_xlabel('Step')
ax.set_ylabel('Win Rate')
ax.set_title('Win Rate Comparison')
ax.legend(loc='upper left')  # 将图例放在左上角
plt.tight_layout()

# 保存图片到同一级目录
plt.savefig('1c3s5z.png', dpi=300, bbox_inches='tight')
# plt.show()