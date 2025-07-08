---
layout:     post
title:      利用swift和mindie在多卡NPU上微调、推理、部署大模型
subtitle:   解决swift 2.5不能NPU多卡推理和部署大模型的问题
date:       2024-9-28
author:     Klaus
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - swift2.5
    - mindie
    - 大模型微调
---

## 前言

本文针对 [swift 2.5](https://github.com/modelscope/swift) 不可以进行NPU**多卡**`推理`和`部署`提出解决方案

10月下旬的 swift 3.0 可能会有NPU多卡推理和训练的方法

解决思路：

- 通过 `swift sft` 对原始模型进行微调(这一步可以多卡)
- `swift infer` 和 `swift deploy` 中有 `--merge_lora` 参数，设为 true。运行其中一个，会得出拼接后的模型参数
- 通过mindie框架对拼接后的模型参数进行推理和部署

微调框架：[swift 2.5](https://github.com/modelscope/swift)
推理和部署框架：[mindie 1.0rc2](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
LLM：[Qwen1.5系列](https://modelscope.cn/organization/qwen?tab=model)

## swift 2.5 安装

参考------->[NPU推理&微调大模型实战](https://developer.aliyun.com/article/1503494)

未联网服务器安装swift 2.5，参考------->[打包 conda 环境](https://klaus-duan.github.io/2024/09/04/打包conda环境/)

## swift 文件修改

能联网的机器可以忽略这一步。

由于swift框架在运行时不能直接使用本地大模型且我的服务器不联网，需要提前修改框架中的 `~/swift/swift/llm/utils/model.py`

1.修改`ModelType`类下的模型路径：

```python
# qwen1.5

qwen1half_32b_chat = '/data2/dxc/LLMs/Qwen1.5-32B-Chat'
qwen1half_32b_chat_w8a8 = '/data2/dxc/LLMs/Qwen1.5-32B-Chat_w8a8'
qwen1half_72b_chat = '/data2/dxc/LLMs/Qwen1.5-72B-Chat'
qwen1half_72b_chat_w8a16_fast = '/data2/dxc/LLMs/Qwen1.5-72B-Chat_w8a16_fast'
```
2.参考 model.py 中 `@register_model`的同类型模型的写法，写自己的@register_model，例如：

```python
@register_model(
    	ModelType.qwen1half_32b_chat_w8a8,
    	'/data2/dxc/LLMs/Qwen1.5-32B-Chat_w8a8',
    	LoRATM.llama,
    	TemplateType.qwen,
    	support_flash_attn=True,
    	support_vllm=False,
    	support_lmdeploy=False,
    	support_megatron=False,
    	requires=['transformers>=4.37'])
```

3.删掉所有其他的 @register\_model ，后续执行过程中如果还有不联网导致的 register\_model 失败，找到 register\_model 继续删。

进入`~/swift`
## swift sft

多卡NPU大模型微调（默认方式为lora）

```shell
export NPROC_PER_NODE=8 \
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
export HCCL_SOME_VARIABLE=value \
export OMP_NUM_THREADS=192
swift sft \
    --model_type '{你的模型路径}' \
    --dataset '{你的训练集路径}' \
    --val_dataset '{你的测试集路径}' \
    --num_train_epochs 10 \
    --sft_type lora \
    --output_dir output \
    --ddp_backend='hccl' \
    --deepspeed default-zero3 \
    --batch_size 1 \
```

如果没有对输出模型的路径进行设置，默认会在原始模型路径内生成检查点文件夹，例如：`v0-20240924-105443/checkpoint-110/`

## swift infer 

swift 2.5 暂时不支持多卡NPU，实际只用到里面的merge_lora功能

```shell
export NPROC_PER_NODE=8 \
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
export HCCL_SOME_VARIABLE=value
swift infer \
    --model_type '{原始模型路径}' \
    --load_args_from_ckpt_dir true \
    --ckpt_dir '{原始模型路径}/v1-20240925-151141/checkpoint-90/' \
    --load_dataset_config true \
    --tensor_parallel_size 8 \
    --merge_lora true \
    --dataset '你的输入文件'
```

会先处理merge_lora的工作，最终的推理程序会因为单卡内存不足中断。

在检查点文件夹内会出现一个类似名为`v0-20240924-105443/checkpoint-110-merged/`的文件夹，里面为拼接好的权重。

## mindie进行后续的推理和部署

我用的`Qwen1.5系列模型`首先要将`v0-20240924-105443/checkpoint-110-merged/`中的`config.json`的`"sliding_window"`参数修改为与原始模型相同的数。未修改时为`null`，运行时会报错。

其他大模型最好也比较一下初始模型和微调模型的`config.json`,把微调模型的参数改成与原始模型相同。

后续即可对模型进行推理和部署。模型路径为`checkpoint-110-merged`
