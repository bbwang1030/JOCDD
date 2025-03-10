import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

aa = np.array([[43,2,0],[5,45,1],[2,3,49]])
ax = sns.heatmap(aa, annot=True, cmap='Blues')

# 数据生成
np.random.seed(2)

rate = [0]
# window_size = [100,300,500,700,900,1100,1300,1500]
window_size = [250,500,750,1000,1250]
eta = 1/500

plt.rcParams['figure.figsize'] = (10, 8.8)
linestyles = [ '-.', '--', ':','-.', '--','-','-',':']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#FFB6C1', '#9acd32', '#eee8aa', '#8470ff', '#625b57', '#87cefa' ,'#008000','#ff0000']

models = {
    "Lasso": 'Lasso',
    "MLP": 'MLP',
    "DT": 'DT',
    "Bagging": 'Bagging',
    "OSELM": 'OSELM',
    "GB": 'GB',
    "Linear":'Linear',
    "JOCDD": 'JOCDD'
}

F1score = np.ones((len(rate), len(window_size), 8)) * (-1)
for num_r,r in enumerate(rate):
    for num_window,window in enumerate(window_size):
        # shift_path = path + shift + '/'
        sum_power = []
        file = f'out_simu_drift_detection_results_summary_rate_{r:.3f}___{window:.2f}_{eta:.4f}.xlsx'
        # 创建 CSV 读取器
        df = pd.read_excel(file, sheet_name='Model Results')
        # print(df)
        f1_scores = df['F1_Score'].tolist()
        # 逐行读取数据
        for row in f1_scores:
            F1score[num_r, num_window, :] = f1_scores
        print(F1score)

Time = np.ones((len(rate), len(window_size), 8)) * (-1)
for num_r,r in enumerate(rate):
    for num_window,window in enumerate(window_size):
        # shift_path = path + shift + '/'
        sum_power = []
        file = f'out_simu_drift_detection_results_summary_rate_{r:.3f}___{window:.2f}_{eta:.4f}.xlsx'
        # 创建 CSV 读取器
        df = pd.read_excel(file, sheet_name='Model Results')
        # print(df)
        time = df['time'].tolist()
        # 逐行读取数据
        for row in time:
            Time[num_r, num_window, :] = time
        # print(F1score)


model_name = ["Lasso", "MLP", "DT","Bagging","OSELM","GB", "Linear","JOCDD"]
window_name = ["250", "500", "750", "1000", "1250"] # 自变量x

# 因变量，这里用来表示成功率的分布
Success_rate = np.array(F1score[0,:,:])

fig, ax = plt.subplots()
im = ax.imshow((Success_rate).T, cmap = 'summer' , interpolation = 'nearest')

# 设置刻度位置
ax.set_xticks(np.arange(len(window_name)))
ax.set_yticks(np.arange(len(model_name)))
# 设置刻度标签
ax.set_xticklabels(window_name, fontsize="16")
ax.set_yticklabels(model_name, fontsize="16")

# 标注每一个因变量的数值到对应位置
for i in range(len(window_name)):
    for j in range(len(model_name)):
        text = ax.text(i, j, '%.2f'%(Success_rate[i, j]),
                       ha="center", va="center", color="b", fontsize="14")

ax.set_title("Matrix of F1-score", fontsize="16")
# fig.tight_layout()
plt.show()

plt.rcParams['figure.figsize'] = (4, 4)
#对rate调参
for num,(dr_idx, dr) in enumerate(models.items()):
    # print(colors[num])
    plt.plot(np.array(window_size), Time[0,:, num],  color=colors[num], marker=markers[num],
             linestyle=linestyles[num], label="%s" % dr)
# plt.axhline(y=sign_level, color='k')
plt.xlabel('Window size')
# plt.ylabel('Type I Error')
plt.ylabel('Time (s)')
plt.xticks(ticks=np.array(window_size))
plt.yticks(ticks=np.linspace(0, 110, 12))
plt.savefig("Time_window_size_comp_noleg.png" , bbox_inches='tight')
plt.legend(loc=2, prop={'size': 8}, ncol=2)
plt.savefig("Time_window_size_comp.png" , bbox_inches='tight')

plt.show()


