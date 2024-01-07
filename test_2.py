
##  picture Binarized Objective Function and Objective Function figure

import torch

torch.manual_seed(1)

import gpytorch
import botorch

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from scipy.optimize import minimize

import scipy.io as sio
from scipy.io import savemat, loadmat

from tqdm.notebook import tqdm
import warnings

import matlab.engine
eng = matlab.engine.start_matlab("-nojvm -nodesktop -nosplash -python")

plt.style.use("bmh")
THRESHOLD = 91.2    # 拟合度阈值

def y_2d(yID, uID, x):
    yID = matlab.double(yID.tolist())  # 转换为matlab可以识别的数据类型
    uID = matlab.double(uID.tolist())  # 同上
    x0 = matlab.double(x[..., 0].tolist()) # 同上
    x1 = matlab.double(x[..., 1].tolist())  # 同上
    y = eng.ModelID_ARX(yID, uID, x0, x1, 1)    # double类型
    return torch.tensor(y).squeeze()    # 返回Tensor类型

#  二值化
def y_2d_bina(yID, uID, x):
    yID = matlab.double(yID.tolist())  # 转换为matlab可以识别的数据类型
    uID = matlab.double(uID.tolist())  # 同上
    x0 = matlab.double(x[..., 0].tolist()) # 同上
    x1 = matlab.double(x[..., 1].tolist())  # 同上
    y = eng.ModelID_ARX(yID, uID, x0, x1, 1)    # double类型
    y = torch.tensor(y, dtype=torch.float64)
    y = torch.where(y >= THRESHOLD, 1, 0)
    return torch.tensor(y).squeeze()    # 返回Tensor类型


# loaded_variables = loadmat('stepModel_data_IO.mat')     # 加载输入和输出数据
# yID = loaded_variables['yID']
# uID = loaded_variables['uID']
#
# lb = 1      # 搜索空间下界
# ub = 10
#
# xs = torch.linspace(lb, ub, 101)
# x1, x2 = torch.meshgrid(xs, xs, indexing="ij")
# xs = torch.vstack((x1.flatten(), x2.flatten())).transpose(-1, -2)
#
# ys = y_2d(yID, uID, xs)
# ys = torch.reshape(ys, (101, 101))
# ys = np.array(ys)
# ys1 = y_2d_bina(yID, uID, xs)
# ys1 = torch.reshape(ys1, (101, 101))
# ys1 = np.array(ys1)
#
# test_d1 = np.linspace(1, 10, 101)
# test_d2 = np.linspace(1, 10, 101)
#
# variables_dict = {'test_d1': test_d1, 'test_d2': test_d2, 'ys': ys, 'ys1': ys1}
#
# savemat(f'test.mat', variables_dict)
loaded_variables = loadmat(f'test.mat')
test_d_1 = np.array(loaded_variables['test_d1'])
test_d_2 = np.array(loaded_variables['test_d2'])
test_d_1, test_d_2 = np.meshgrid(test_d_1, test_d_2)
Ys = np.array(loaded_variables['ys'])
Ys1 = np.array(loaded_variables['ys1'])

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
fig.subplots_adjust(top=0.8)
ax1.plot_surface(test_d_1, test_d_2, Ys, cmap=cm.coolwarm, linewidth=0, alpha=0.45, antialiased=False)
ax1.set_title("Objective Function")
ax1.set_xlabel('parameter na')
ax1.set_ylabel('parameter nb')
ax1.set_zlabel('Objective value')
ax1.patch.set_alpha(0)

ax2 = fig.add_subplot(122, projection='3d')
fig.subplots_adjust(top=0.8)
ax2.plot_surface(test_d_1, test_d_2, Ys1, cmap=cm.coolwarm, linewidth=0, alpha=0.45, antialiased=False)
ax2.set_title("Binarized Objective Function")
ax2.set_xlabel('parameter na')
ax2.set_ylabel('parameter nb')
ax2.set_zlabel('Binarized Objective value')
ax2.patch.set_alpha(0)

plt.savefig('Binarized_Objective.png')
plt.show()
print('1')