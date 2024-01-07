

import torch
import gpytorch
import botorch

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.io import savemat, loadmat
from scipy.integrate import quad
import warnings
from scipy.integrate import IntegrationWarning
import torch.nn.functional as F

from scipy.optimize import minimize
#
# plt.style.use("bmh")
#
#
# SWITCH = 1
# THRESHOLD = 2.4
#
#
# def forrester_1d_multi(x):
#     if SWITCH:
#         noise = torch.tensor(np.random.normal(0, 0.02, size=x.shape), dtype=torch.float32)
#     else:
#         noise = 0
#     y = 2.5 * torch.exp(-((x+2)**2)/(2*1*1)) + 3 * torch.exp(-((x-1)**2)/(2*0.8*0.8))+noise
#     y = torch.where(y >= THRESHOLD, 1, 0)
#     return y.squeeze(-1)
#
#
# def visualize_gp_belief_and_policy(model, likelihood, policy=None, next_x=None, iteration=1):
#     with torch.no_grad():
#         predictive_distribution = likelihood(model(xs))
#         predictive_mean = predictive_distribution.mean
#         predictive_upper, predictive_lower = predictive_distribution.confidence_region()
#
#         if policy is not None:
#             acquisition_score = policy(xs.unsqueeze(1))
#
#     if policy is None:
#         plt.figure(figsize=(8, 3))
#
#         plt.plot(xs, ys, label="objective", c="r")
#         plt.scatter(train_x, train_y, marker="x", c="k", label="observations")
#
#         plt.plot(xs, predictive_mean, label="mean", title=f'{iteration+2} observations')
#         plt.fill_between(
#             xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
#         )
#
#         plt.legend()
#         # plt.title(f'{iteration} new observations')
#         if iteration < 10:
#             plt.savefig(f'figs3/iteration_0{iteration}.png')
#         plt.savefig(f'figs3/iteration_{iteration}.png')
#         plt.show()
#     else:
#         fig, ax = plt.subplots(
#             2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
#         )
#
#         # GP belief
#         ax[0].plot(xs, ys, label="objective", c="r")
#         ax[0].scatter(train_x, train_y, marker="x", c="k", label="observations")
#
#         ax[0].plot(xs, predictive_mean, label="mean")
#         ax[0].fill_between(
#             xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
#         )
#         ax[0].set_title(f'{iteration+2} observations')
#
#         if next_x is not None:
#             ax[0].axvline(next_x.item(), linestyle="dotted", c="k")
#
#         ax[0].legend(framealpha=0.5)
#         ax[0].set_ylabel("predictive")
#         ax[0].patch.set_alpha(0)
#
#         # acquisition score
#         ax[1].plot(xs, acquisition_score, c="g")
#         ax[1].fill_between(xs.flatten(), acquisition_score, 0, color="g", alpha=0.5)
#
#         if next_x is not None:
#             ax[1].axvline(next_x.item(), linestyle="dotted", c="k")
#
#         ax[1].set_ylabel("acquisition score")
#         ax[1].patch.set_alpha(0)
#         if iteration < 10:
#             plt.savefig(f'figs3/iteration_0{iteration}.png')
#         plt.savefig(f'figs3/iteration_{iteration}.png')
#         plt.show()
#
#
# class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
#     _num_outputs = 1
#
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#
# def fit_gp_model(train_x, train_y, noise=1e-4, num_train_iters=500):
#     # declare the GP
#     # noise = 1e-4
#
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     model = GPModel(train_x, train_y, likelihood)
#     model.likelihood.noise = noise
#
#     # train the hyperparameter (the constant)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#
#     model.train()
#     likelihood.train()
#
#     for i in tqdm(range(num_train_iters)):
#         optimizer.zero_grad()
#
#         output = model(train_x)
#         loss = -mll(output, train_y)
#
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     likelihood.eval()
#
#     return model, likelihood
#
#
# bound = 4
#
# xs = torch.linspace(-bound, bound, bound * 100 + 1).unsqueeze(1)
# ys = forrester_1d_multi(xs)
#
# torch.manual_seed(1)
# train_x = torch.rand(size=(1, 1)) * 2 * bound - bound
# train_y = forrester_1d_multi(train_x)
#
# train_x = torch.tensor([[-2.0], [1.0]])
# train_y = forrester_1d_multi(train_x)
#
# print(torch.hstack([train_x, train_y.unsqueeze(1)]))
#
# model, likelihood = fit_gp_model(train_x, train_y)
#
# num_queries = 360       # 240 Iterations
# Fit_noise = 1e-4
#
# for i in range(num_queries):
#     print("iteration", i)
#     print("incumbent", train_x[train_y.argmax()], train_y.max())
#
#     model, likelihood = fit_gp_model(train_x, train_y, Fit_noise)
#
#     policy = botorch.acquisition.analytic.ExpectedImprovement(
#         model, best_f=train_y.max()
#     )
#
#     next_x, acq_val = botorch.optim.optimize_acqf(
#         # policy,
#         # bounds=torch.tensor([[-bound * 1.0], [bound * 1.0]]),
#         # q=1,
#         # num_restarts=20,
#         # raw_samples=50,
#         policy,
#         bounds=torch.tensor([[-bound * 1.0], [bound * 1.0]]),
#         q=1,
#         num_restarts=40,
#         raw_samples=100,
#     )
#
#     visualize_gp_belief_and_policy(model, likelihood, policy, next_x=next_x, iteration=i)
#
#     next_y = forrester_1d_multi(next_x)
#
#     train_x = torch.cat([train_x, next_x])
#     train_y = torch.cat([train_y, next_y])
#
#
# train_X = np.array(train_x)
# train_Y = np.array(train_y)
# print(train_X)
#
# plt.hist(train_X, bins=30, density=True, alpha=0.5, color='b', label='Histogram')
#
# mean_value = np.mean(train_X)
# variance_value = np.var(train_X)
# print(train_X)
# print(mean_value)
# print(variance_value)
# variables_dict = {'train_X': train_X, 'train_Y': train_Y, 'mean': mean_value, 'cov': variance_value}
#
# savemat(f'multiple_variables_1_modify.mat', variables_dict)

loaded_variables = loadmat(f'multiple_variables_1_modify.mat')
train_X = loaded_variables['train_X'].tolist()
print(loaded_variables['train_Y'])
train_X = np.array(train_X).reshape(1, -1).flatten()  # 1-D array

kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
kde.fit(train_X[:, None])
x_vals = np.linspace(-3, 3, 1000)
log_dens = kde.score_samples(x_vals[:, None])
plt.hist(train_X, bins=30, density=True, alpha=0.5, color='b', label='Histogram')
plt.plot(x_vals, np.exp(log_dens), 'r-', label='Kernel Density Estimation')
mean = loaded_variables['mean'].flatten()
cov = loaded_variables['cov'].flatten()
plt.title(f'Histogram distribution and probability distribution of\n '
          f'361 parameter observation points\nh=0.1')

# [-0.5, 0.5]
a, b = 0.3, 1.7
m, n = -2.3, -1.7

warnings.filterwarnings("ignore", category=IntegrationWarning)
pdf = np.exp(kde.score_samples(np.array([[x] for x in x_vals])))
probability, _ = quad(lambda x: np.interp(x, x_vals, pdf), a, b)    #
print(f'Probability of {a} <= X <= {b}: {probability}')
probability, _ = quad(lambda x: np.interp(x, x_vals, pdf), m, n)    #
print(f'Probability of {m} <= X <= {n}: {probability}')


plt.xlabel('Parameter Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()

print(loaded_variables['mean'])
print(loaded_variables['cov'])
