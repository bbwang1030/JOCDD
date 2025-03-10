import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

aa = np.array([[43,2,0],[5,45,1],[2,3,49]])
ax = sns.heatmap(aa, annot=True, cmap='Blues')

# 数据生成
np.random.seed(2)

rate = [0.005]
threshold = 5 # 阈值为窗口大小的1%默认20
eta_list=[1/100,1/200,1/500,1/1000,1/2000,1 / 3000, 1 / 4000, 1 / 5000]
eta_list=[1/5000,1/2000,1/1000,1/500,1/200,1/100]

plt.rcParams['figure.figsize'] = (7, 7)
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

F1score = np.ones((len(rate), len(eta_list), 8)) * (-1)
for num_r,r in enumerate(rate):
    for num_eta,eta in enumerate(eta_list):
        # shift_path = path + shift + '/'
        sum_power = []
        file = f'out_simu_drift_detection_results_summary_rate_{r:.3f}_{threshold:.2f}_{eta:.4f}.xlsx'
        # 创建 CSV 读取器
        df = pd.read_excel(file, sheet_name='Model Results')
        # print(df)
        f1_scores = df['F1_Score'].tolist()
        # 逐行读取数据
        for row in f1_scores:
            F1score[num_r, num_eta, :] = f1_scores
        print(F1score)

Time = np.ones((len(rate), len(eta_list), 8)) * (-1)
for num_r,r in enumerate(rate):
    for num_eta,eta in enumerate(eta_list):
        # shift_path = path + shift + '/'
        sum_power = []
        file = f'out_simu_drift_detection_results_summary_rate_{r:.3f}_{threshold:.2f}_{eta:.4f}.xlsx'
        # 创建 CSV 读取器
        df = pd.read_excel(file, sheet_name='Model Results')
        # print(df)
        time = df['time'].tolist()
        # 逐行读取数据
        for row in time:
            Time[num_r, num_eta, :] = time
        # print(F1score)


model_name = ["Lasso", "MLP", "DT","Bagging","OSELM","GB", "Linear","JOCDD"]
eta_name = ["1/5000","1/2000","1/1000","1/500","1/200","1/100"] # 自变量x

# 因变量，这里用来表示成功率的分布
Success_rate = np.array(F1score[0,:,:])

fig, ax = plt.subplots()
im = ax.imshow((Success_rate).T, cmap = 'summer' , interpolation = 'nearest')

# 设置刻度位置
ax.set_xticks(np.arange(len(eta_name)))
ax.set_yticks(np.arange(len(model_name)))
# 设置刻度标签
ax.set_xticklabels(eta_name, fontsize="16")
ax.set_yticklabels(model_name, fontsize="16")

# 标注每一个因变量的数值到对应位置
for i in range(len(eta_name)):
    for j in range(len(model_name)):
        text = ax.text(i, j, '%.2f'%(Success_rate[i, j]),
                       ha="center", va="center", color="b", fontsize="14")

# ax.set_title("Matrix of F1-score", fontsize="16")
# fig.tight_layout()
plt.show()

plt.rcParams['figure.figsize'] = (4, 4)
#对rate调参
# for num,(dr_idx, dr) in enumerate(models.items()):
#     # print(colors[num])
num=7
dr="JOCDD"
fig, ax1 = plt.subplots()
plt.plot(range(len(eta_list)), Time[0,:, num],  color='g',label="Time")
ax1.set_xlabel('Decrement rate')
ax1.set_ylabel('Time (s)', color='g')
ax1.set_xticks(ticks=range(len(eta_list)),labels=eta_name)
ax1.set_yticks(ticks=np.linspace(0, 10, 11))
plt.legend(loc=4, prop={'size': 10}, ncol=2)
ax2 = ax1.twinx()
print(Success_rate.shape)
plt.plot(range(len(eta_list)), Success_rate[:, num], 'r', label="F1-score" )
ax2.set_ylabel('F1-score', color='r')
ax2.set_yticks(ticks=np.linspace(0, 1, 11))
plt.savefig("Time_eta_comp_noleg.png" , bbox_inches='tight')
plt.legend(loc=1, prop={'size': 10}, ncol=2)
plt.savefig("Time_eta_comp.png" , bbox_inches='tight')
plt.show()


