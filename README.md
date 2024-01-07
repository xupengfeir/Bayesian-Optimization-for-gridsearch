### 运行环境：matlab和python3

1、安装Anaconda3/Python3.9和matlab2021b

2、按照https://github.com/pytorch/botorch 安装botorch、pytorch和gpytorch。

```python
conda install botorch -c pytorch -c gpytorch -c conda-forge
```

3、安装matlab engine。

```bash
cd matlabroot\extern\engines\python
python setup.py install
```

4、文件

```text
1、.mat：数据
2、.m .mdl模型文件
3、batesopt_2d_step_arx.py：ARX模型参数估计程序
   test_2.py：ARX模型目标函数和二值化目标函数对比图
```







