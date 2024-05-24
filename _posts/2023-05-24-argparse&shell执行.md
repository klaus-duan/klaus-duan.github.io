---
layout:     post
title:      代码执行须知
subtitle:   argparse & shell 的简单介绍
date:       2023-05-24
author:     Klaus
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - shell
    - 脚本
---

## 前言

一种可以在主代码之外设置参数的方法。通过`argparse`包添加参数，这样就可以在执行代码时不用再去找主代码的参数。后续加上`shell`可以让命令执行更方便快捷。

## argparse使用

### 导入argparse

	import argparse

### 创建对象

	parser = argparse.ArgumentParser(description='A simple script with argparse')

description为对象的描述内容，可以通过`python script.py -h` 或 `python script.py --help` 指令查看到。通常建议一个脚本里只设置一个argparse.ArgumentParser对象。

例：

	(pytorch) klaus_d.@192 ~ % python /Users/klaus_d./Desktop/script.py -h
	usage: script.py [-h] [--param1 PARAM1] [--param2 PARAM2]
	
	A simple script with argparse      -->对象的argparse描述
	
	options:
	  -h, --help       show this help message and exit
	  --param1 PARAM1  An integer parameter
	  --param2 PARAM2  A string parameter

### 添加参数

上面我们创建了一个名为`parser`的对象，下面来通过`parser.add_argument`向脚本中添加参数。

以下是一个例子：

	parser.add_argument('--param1', type=float, help='An integer parameter')
	parser.add_argument('--param2', type=str, help='A string parameter')

以上通过`parser.add_argument`我们向脚本里添加了两个参数，一个名为`--param1`，数据类型为`float`；一个名为`--param2`，数据类型为`str`。

`parser.add_argument`中还可以设置其他参数，具体如下：

- `name_or_flags`：命令行参数的名字或者列表。例如，--myparam 或 -p。
- `action`：当参数在命令行中出现时的动作。默认是 store，表示存储参数的值。其他可能的值包括 store_const、store_true、store_false、append、append_const、count 和 help。
- `nargs`：命令行参数应该消耗的命令行参数数量。它可以是一个整数或者 ?、*、+、argparse.REMAINDER。
- `const`：一些 action 和 nargs 选项需要的常数值。
- `default`：如果命令行中没有出现参数，那么使用的默认值。
- `type`：一个函数，它接受一个字符串参数，并返回参数的类型。默认是 str。
- `choices`：参数的值应该是这个列表中的一个。
- `required`：是否是必需的命令行参数（默认是 False）。
- `help`：参数的简短描述。
- `metavar`：在使用或者帮助信息中使用的参数值示例。
- `dest`：添加到返回的对象中的属性的名字。

这是一个复杂一点的例子：

	parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')
                        
### 解析命令行参数

	args = parser.parse_args()

## 通过terminal直接执行脚本

### 示例代码

	# 这个代码将会存储在一个名为 script.py 的文件里，以便后续使用
	import argparse
	
	parser = argparse.ArgumentParser(description='A simple script with argparse',)
	
	parser.add_argument('--param1', type=float, help='An integer parameter')
	parser.add_argument('--param2', type=str, help='A string parameter')
	
	args = parser.parse_args()
	
	print(f"Received param1 with value {args.param1}")
	print(f"Received param2 with value {args.param2}")
	
### 执行指令

打开terminal，输入`指令`

	python -u script.py --param1 123 --param2 "hello" 
	# 加上 -u 可以立即看到脚本的输出

输出

	(pytorch) klaus_d.@192 ~ % python -u /Users/klaus_d./Desktop/script.py --param1 123 --param2 "hello"
	Received param1 with value 123.0
	Received param2 with value hello
	# 后两行即为我们要打印的内容，可以与代码的print行进行对比
	
## 通过shell文件执行脚本

在我们上面的例子中，执行脚本需要手动输入参数，如果遇到参数量较多且需要重复输入的任务，可以编写shell文件，将要执行的文件和参数输入进去，做修改时只需要改变shell文件里的内容。

### shell编写

	# 此文件保存为 run.sh
	python /Users/klaus_d./Desktop/script.py \
	    --param1 1314 \
	    --param2 'hello'     

shell编写时，`\`后一定不要有任何内容，空格和换行也不行

### shell执行

打开terminal，输入`bash run.sh`

	(pytorch) klaus_d.@192 ~ % bash /Users/klaus_d./Desktop/run.sh

输出

	(pytorch) klaus_d.@192 ~ % bash /Users/klaus_d./Desktop/run.sh
	Received param1 with value 1314.0
	Received param2 with value hello1

## 执行多个脚本和使用共用参数

### 使用共用值

当几个参数共用一个值时，可以设置一个共用值

	abc='1314'   # 定义共用值abc，“=”两边不要有空格，要放在参数前，否则会报错
	python /Users/klaus_d./Desktop/script.py \
	    --param1 $abc \
	    --param2 $abc
	    
执行

> bash run.sh

输出

	Received param1 with value 1314.0
	Received param2 with value 1314

### 一个shell文件执行多个脚本

建立一个新的脚本文件`sp.py`

	import argparse
	
	parser = argparse.ArgumentParser(description='A simple script with argparse',)
	
	parser.add_argument('--p1', type=float, help='An integer parameter')
	parser.add_argument('-p2', type=str, help='A string parameter')
	
	args = parser.parse_args()
	
	print(f"返回 param1 with value {args.p1}")
	print(f"返回 param2 with value {args.p2}")
	
编写shell文件

	abc='1314'   # 定义共用值abc
	python /Users/klaus_d./Desktop/script.py \
	    --param1 $abc \
	    --param2 $abc  # 此文件保存为 run.sh
	
	python /Users/klaus_d./Desktop/sp.py \
	    --p1 2 \
	    -p2 'hello2'
	
	python /Users/klaus_d./Desktop/script.py \
	    --param1 3 \
	    --param2 'hello3'
	
	python /Users/klaus_d./Desktop/sp.py \
	    --p1 $abc \
	    -p2 'hello4'
	    
执行

> bash run.sh

结果

	Received param1 with value 1314.0
	Received param2 with value 1314
	返回 param1 with value 2.0
	返回 param2 with value hello2
	Received param1 with value 3.0
	Received param2 with value hello3
	返回 param1 with value 1314.0
	返回 param2 with value hello4
 
#### 在shell脚本中运行多个脚本文件是完全可以的，但是需要注意一些事项：

- 确保每个脚本都能独立运行，不依赖于其他脚本的运行状态或结果。
- 如果脚本之间有依赖关系，需要正确地管理这些依赖关系，确保脚本按照正确的顺序执行。
- 如果脚本的执行时间较长，可能需要考虑并行执行以提高效率。但是，这可能会增加脚本的复杂性，并可能需要额外的同步机制。
- 考虑错误处理。如果一个脚本失败，你可能需要决定是否继续执行其他脚本，或者如何通知用户。
