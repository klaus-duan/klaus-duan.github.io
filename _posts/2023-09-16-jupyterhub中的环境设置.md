---
layout:     post
title:      jupyterhub中的环境设置
subtitle:   有关conda环境的建立删除，以及pip路径的设置
date:       2023-09-16
author:     Klaus
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - jupyterhub
    - conda
---

## 前言

从github上clone的一些代码，运行之前遇到许多问题。比较大的是即使激活了相关环境，但是pip install 的地址却还是初始地址，导致匹配问题。解决方法是设置pip路径，我写的这种方法是临时路径，每次都要设置一次。永久路径的设置方法还没有试过。`(输出样式有些问题，比如代码的两个-会被显示成一个-）`

## 删除conda环境和对应的kernel

### 删除conda环境

- 退出要删除的环境

	> conda deactivate

- 删除环境

	> conda env remove -n CLNER
	
	这个命令会删除名为 CLNER 的Conda环境及其所有依赖。
	
### 删除 Jupyter Kernel

- 找到kernel的目录
	
	> jupyter kernelspec list
	
- 删除特定kernel
	
	> rm -rf ~/.local/share/jupyter/kernels/CLNER

## 创建conda环境

### 创建

> conda create -n CLNER python=3.10  # 或选择其他版本

### 激活

> conda activate CLNER

## 设置pip路径

### 检查pip路径

> which pip 

### 设置pip路径 `此过程为临时设置，下次需要时还要再次重复这个过程`
当pip路径与conda环境路径不同（比如使用的是系统默认路径），需要手动设置路径。

1. 打开 Bash 配置文件，通常是 ~/.bashrc 或 ~/.bash_profile。

	> nano ~/.bashrc 
	
2. 在文件末尾添加以下行，以更新 PATH 变量：

	> export PATH=/home/lihan/miniconda3/envs/CLNER/bin:$PATH
	
3. 保存并关闭文件。

4. 为了使更改生效，运行以下命令：

	> source ~/.bashrc

### 检查pip路径
	
> which pip 

## 安装requirements.txt

在确定pip路径在当下conda环境之后，运行安装指令

切换文件夹

> cd ~/path/to

安装

> pip install -r requirements.txt

## 安装kernel

在jupyter建立文件时，当需要使用新建的conda环境作为内核时，有以下设置。

1. 激活Conda环境：

	> conda activate env_name

2. 安装IPython kernel：

	> pip install ipykernel
	
	或使用conda安装
	
	> conda install -c conda-forge ipykernel
	
3. 添加新的Jupyter kernel:

	> python -m ipykernel install --user --name `env_name` --display-name "`display_name`"

这样新的kernel就会出现在你的jupyter的内核列表里。
