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
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
import warnings
from scipy.integrate import IntegrationWarning

from tqdm.notebook import tqdm
import warnings
import seaborn as sns

import matlab.engine
eng = matlab.engine.start_matlab("-nojvm -nodesktop -nosplash -python")

plt.style.use("bmh")

def visualize_progress_and_policy(policy, next_x=None, iteration=1):
    with torch.no_grad():
        acquisition_score = policy(xs.unsqueeze(1))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # ground truth
    c = ax[0].imshow(ys.reshape(101, 101).T, cmap=cm.coolwarm, origin="lower", extent=[lb, ub, lb, ub])
    ax[0].set_xlabel(r"$na$", fontsize=20)
    ax[0].set_ylabel(r"$nb$", fontsize=20)
    plt.colorbar(c, ax=ax[0])

    ax[0].scatter(train_x[..., 0], train_x[..., 1], marker="x", c="k", label="observations")

    ax[0].set_title(f"{iteration+3} observations")

    # acquisition score
    c = ax[1].imshow(
        acquisition_score.reshape(101, 101).T, cmap=cm.coolwarm, origin="lower", extent=[lb, ub, lb, ub]
    )
    ax[1].set_xlabel(r"$na$", fontsize=20)
    plt.colorbar(c, ax=ax[1])

    ax[1].scatter(
        train_x[..., 0], train_x[..., 1], marker="x", c="k", label="observations"
    )
    if next_x is not None:
        ax[1].scatter(
            next_x[..., 0], next_x[..., 1], c="r", marker="*", s=500, label="next query"
        )

    ax[1].legend()
    ax[1].set_title("acquisition score")
    if iteration < 10:
        plt.savefig(f'figs/iteration_0{iteration}.png')
    plt.savefig(f'figs/iteration_{iteration}.png')
    plt.show()


class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        multi_normal = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return multi_normal

    def mean_covar(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return mean_x, covar_x

def fit_gp_model(train_x, train_y, num_train_iters=500):
    # declare the GP
    noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    multi_normal = model.forward(train_x)
    mean, covar = model.mean_covar(train_x)

    model.likelihood.noise = noise

    # train the hyperparameter (the constant)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for i in range(num_train_iters):
        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood, multi_normal, mean, covar


###  multi-Dimension 2-D

THRESHOLD = 91.2

#  二值化后的目标函数
def y_2d_bina(yID, uID, x):
    yID = matlab.double(yID.tolist())  # 转换为matlab可以识别的数据类型
    uID = matlab.double(uID.tolist())  # 同上
    x0 = matlab.double(x[..., 0].tolist()) # 同上
    x1 = matlab.double(x[..., 1].tolist())  # 同上
    y = eng.ModelID_ARX(yID, uID, x0, x1, 1)    # double类型
    y = torch.tensor(y, dtype=torch.float64)
    y = torch.where(y >= THRESHOLD, 1, 0)
    return torch.tensor(y).squeeze()    # 返回Tensor类型


loaded_variables = loadmat('stepModel_data_IO.mat')     # 加载输入和输出数据
yID = loaded_variables['yID']
uID = loaded_variables['uID']

lb = 1      # 搜索空间下界
ub = 10     # 搜索空间上界
num_queries = 360        # 迭代次数

bounds = torch.tensor([[lb, lb], [ub, ub]], dtype=torch.float)

xs = torch.linspace(lb, ub, 101)
x1, x2 = torch.meshgrid(xs, xs, indexing="ij")
xs = torch.vstack((x1.flatten(), x2.flatten())).transpose(-1, -2)
ys = y_2d_bina(yID, uID, xs)

# 设置3个初始观测点[1,1],[2,2],[6,6]
train_x = torch.tensor(
    [
        [1.0, 1.0], [1.0, 2.0], [4.0, 8.0]
    ]
)

train_y = y_2d_bina(yID, uID, train_x)

epsilon = 0.1
for i in range(num_queries):
    print("iteration", i)
    print("incumbent", train_x[train_y.argmax()], train_y.max())

    model, likelihood, multi_normal, mean, covar = fit_gp_model(train_x, train_y)

    policy = botorch.acquisition.analytic.ProbabilityOfImprovement(
        model, best_f=train_y.max() + epsilon
    )

    next_x, acq_val = botorch.optim.optimize_acqf(
        policy,
        bounds=bounds,
        q=1,
        num_restarts=40,
        raw_samples=100,
    )

    visualize_progress_and_policy(policy, next_x=next_x, iteration=i)

    next_y = y_2d_bina(yID, uID, next_x)
    next_y = torch.unsqueeze(next_y, dim=0)     # 将零维张量扩展成一维张量

    train_x = torch.cat([train_x, next_x])
    train_y = torch.cat([train_y, next_y])


train_X = np.array(train_x)
train_Y = np.array(train_y)
print(train_X)
print(train_Y)
# 计算均值和方差
mean_value = np.mean(train_X)
variance_value = np.var(train_X)
print(train_X)
print(mean_value)
print(variance_value)
variables_dict = {'train_X': train_X, 'train_Y': train_Y, 'mean': mean_value, 'cov': variance_value}
# 保存多个变量到MAT文件
savemat(f'multiple_variables_arx.mat', variables_dict)



loaded_variables = loadmat(f'multiple_variables_arx.mat')
train_X = loaded_variables['train_X'].tolist()      # 两个变量 后面分别计算核密度估计
print(loaded_variables['train_Y'])
train_X = np.array(train_X)  # 2-D array
# 1、二维分布
# 绘制核密度估计图（二维）
kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
kde.fit(train_X)

# 生成网格坐标
x_grid, y_grid = np.meshgrid(np.linspace(0, 12, 100), np.linspace(0, 12, 100))
grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# 计算在网格上的概率密度
prob_density = np.exp(kde.score_samples(grid_coords))
prob_density = prob_density.reshape(x_grid.shape)
# 创建3D图
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
# 绘制3D表面图
ax.plot_surface(x_grid, y_grid, prob_density, cmap='coolwarm', edgecolor='k')

# 设置图形标签
ax.set_xlabel('na')
ax.set_ylabel('nb')
ax.set_zlabel('Probability Density')
ax.set_title('Two-dimensional parametric probability distribution')
ax.patch.set_alpha(0)

## 2、参数na的分布
kde_1 = KernelDensity(bandwidth=0.15, kernel='gaussian')
kde_1.fit(train_X[:, 0].reshape(-1, 1))
x_vals_1 = np.linspace(0, 12, 1000)
log_dens_1 = kde_1.score_samples(x_vals_1[:, None])
fig, ax1 = plt.subplots(figsize=(7, 6))
ax1.set_facecolor('none')
ax1.hist(train_X[:, 0].reshape(-1, 1), bins=30, density=True, alpha=0.5, color='b', label='Histogram')
ax1.plot(x_vals_1, np.exp(log_dens_1), 'r-', label='Kernel Density Estimation')
ax1.set_ylabel('Probability Density')
ax1.set_title(f'Histogram distribution and probability distribution of\n '
          f'362 parameter observation points for parameter na\nh=0.15')
plt.legend()


## 3、参数nb的分布
kde_2 = KernelDensity(bandwidth=0.15, kernel='gaussian')
kde_2.fit(train_X[:, 1].reshape(-1, 1))
x_vals_2 = np.linspace(0, 12, 1000)
log_dens_2 = kde_2.score_samples(x_vals_2[:, None])
fig, ax2 = plt.subplots(figsize=(7, 6))
ax2.set_facecolor('none')
ax2.hist(train_X[:, 1].reshape(-1, 1), bins=30, density=True, alpha=0.5, color='b', label='Histogram')
ax2.plot(x_vals_2, np.exp(log_dens_2), 'r-', label='Kernel Density Estimation')
ax2.set_ylabel('Probability Density')
ax2.set_title(f'Histogram distribution and probability distribution of\n'
          f'362 parameter observation points for parameter nb\nh=0.15')
plt.legend()

plt.show()


## 计算区间概率
a, b = 5, 10
m, n = 2, 7

warnings.filterwarnings("ignore", category=IntegrationWarning)
pdf = np.exp(kde_1.score_samples(np.array([[x] for x in x_vals_1])))
probability, _ = quad(lambda x: np.interp(x, x_vals_1, pdf), a, b)    #
print(f'Probability of {a} <= X <= {b}: {probability}')
pdf = np.exp(kde_2.score_samples(np.array([[x] for x in x_vals_2])))
probability, _ = quad(lambda x: np.interp(x, x_vals_2, pdf), m, n)    #
print(f'Probability of {m} <= X <= {n}: {probability}')


eng.exit()
