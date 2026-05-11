---
layout:     post
title:      许可证识别模型训练
subtitle:   基于bert的许可证多标签分类识别
date:       2025-10-02
author:     Klaus
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - bert
    - 多标签分类
---

## 前言

- 训练`多标签文本分类`模型，用于识别许可证种类。
- 基础模型为`bert`，训练之后识别准确率85%。

## 数据预处理

**Qwen-max**：针对`LICENSE`或`README.md`中的文本，删除和许可证信息无关的内容，将许可证原文缩短。

````markdown
# system prompt
你是一个专门处理软件许可证文本的助手。你的任务是从用户输入的文本中**提取许可证信息**并输出，允许在特定情况下使用许可证名称代替原文。

规则：
1. 删除所有与许可证无关的内容（如项目介绍、安装步骤、使用方法、配置说明、贡献指南、行为准则、作者列表、联系方式、徽章链接、赞助信息、测试状态等）。
2. 对于许可证相关内容，按以下优先级处理：
   - **如果文本中包含完整的标准开源许可证全文（如 MIT、Apache 2.0、GPL、BSD 等），且你能明确识别出该许可证的名称**，则输出该许可证的标准名称（例如：“MIT License”、“Apache License 2.0”），**无需输出全文**。
   - **如果文本中包含对多个许可证的引用或选择条款**（如 “either...or”、“both...and”、“licensed under...or”），则输出这些许可证的名称以及相关选择/引用语句的原文。
   - **如果文本中包含非标准的版权声明、自定义授权条款、免责声明、指向其他许可证文件的引用**，则原样输出这些内容（不改变措辞、标点、换行）。
   - **如果文本中只包含许可证名称而没有全文**，则直接输出该名称。
3. 当输出许可证名称时，使用常见的标准写法（如 “MIT License”、“Apache License, Version 2.0”）。
4. 绝不添加任何额外的解释、总结、序号或符号。
5. 如果文本中没有找到任何许可证信息，输出空字符串。

示例：
```
输入：
"Spack is a multi-platform package manager. License: Spack is distributed under the terms of both the MIT license and the Apache License (Version 2.0). Users may choose either license. For installation, run git clone ..."

输出（识别出标准许可证名称，输出名称与选择语句）：
"MIT License and Apache License 2.0. Users may choose either license."

输入：
"Copyright (c) 2024 Roman Krasilnikov. Permission is hereby granted...（MIT 全文）... Portions of this project are licensed under the Apache License, Version 2.0. See the LICENSE-APACHE file."

输出（MIT 全文被识别为 MIT License，输出名称；Apache 部分输出引用原文）：
"MIT License
Portions of this project are licensed under the Apache License, Version 2.0. See the LICENSE-APACHE file for more details."
```
````

处理示例：

````markdown
# LICENSE[处理前]

```
Copyright (c) 2024 Roman Krasilnikov

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Portions of this project are licensed under the Apache License, Version 2.0. See the LICENSE-APACHE file for more details.
```

# LICENSE[Qwen-max提取后]

```
MIT license
Portions of this project are licensed under the Apache License, Version 2.0. See the LICENSE-APACHE file for more details.
```
````

````markdown
# README.md[处理前]

```
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="proprietary/images/OpenFrontLogoDark.svg">
    <source media="(prefers-color-scheme: light)" srcset="proprietary/images/OpenFrontLogo.svg">
    <img src="proprietary/images/OpenFrontLogo.svg" alt="OpenFrontIO Logo" width="300">
  </picture>
</p>

[OpenFront.io](https://openfront.io/) is an online real-time strategy game focused on territorial control and alliance building. Players compete to expand their territory, build structures, and form strategic alliances in various maps based on real-world geography.

This is a fork/rewrite of WarFront.io. Credit to https://github.com/WarFrontIO.

![CI](https://github.com/openfrontio/OpenFrontIO/actions/workflows/ci.yml/badge.svg)
[![Crowdin](https://badges.crowdin.net/openfront-mls/localized.svg)](https://crowdin.com/project/openfront-mls)
[![CLA assistant](https://cla-assistant.io/readme/badge/openfrontio/OpenFrontIO)](https://cla-assistant.io/openfrontio/OpenFrontIO)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Assets: CC BY-SA 4.0](https://img.shields.io/badge/Assets-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

## License

OpenFront source code is licensed under the **GNU Affero General Public License v3.0**

Current copyright notices appear in:

- Footer: "© OpenFront and Contributors"
- Loading screen: "© OpenFront and Contributors"

Modified versions must preserve these notices in reasonably visible locations.

See the [LICENSE](LICENSE) for complete requirements.

For asset licensing, see [LICENSE-ASSETS](LICENSE-ASSETS).  
For license history, see [LICENSING.md](LICENSING.md).

## 🌟 Features
...

### Final Notes

Remember that maintaining this project requires significant effort. The maintainer appreciates your contributions but must prioritize long-term project health and stability. Not all contributions will be accepted, and that's okay.

Thank you for helping make OpenFront better!
```

# README.md[Qwen-max提取后]

```
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Assets: CC BY-SA 4.0](https://img.shields.io/badge/Assets-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

## License

OpenFront source code is licensed under the **GNU Affero General Public License v3.0**

Current copyright notices appear in:

- Footer: "© OpenFront and Contributors"
- Loading screen: "© OpenFront and Contributors"

Modified versions must preserve these notices in reasonably visible locations.

See the [LICENSE](LICENSE) for complete requirements.

For asset licensing, see [LICENSE-ASSETS](LICENSE-ASSETS).  
For license history, see [LICENSING.md](LICENSING.md).
```
````

## bert识别许可证种类

常用license列表：

```text
[MIT, Apache-2.0, GPL-3.0, LGPL-3.0, BSD-3-Clause, BSD-2-Clause, AGPL-3.0, MPL-2.0, ISC, Unlicense, CC0-1.0, EPL-2.0, MS-PL, Zlib, OSL-3.0, Artistic-2.0, EUPL-1.2, NCSA, PostgreSQL, Sleepycat]
```

标签示例：

```text
[0,0,0,1,0,1,...]
```

训练代码示例：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 20  # 许可证类别数量（根据你的标签列表长度）
label_names = ["MIT", "Apache-2.0", "GPL-3.0", ...]  # 完整列表
max_len = 512
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

# 加载预训练模型和分词器（使用英文 bert，因为许可证文本为英文）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels,
    problem_type="multi_label_classification"  # 关键
).to(device)

# ========== 示例数据 ==========

sentences = [  # 文本列表
    "MIT license",
    "Apache License 2.0 and MIT",
    "Portions of this project are licensed under the Apache License, Version 2.0.",
    # ... 更多
]
labels = [  # 多热标签，形状 (样本数, num_labels)，0/1
    [1, 0, 0, ...],  # MIT
    [1, 1, 0, ...],  # MIT + Apache-2.0
    [0, 1, 0, ...],  # Apache-2.0
    # ...
]
labels = torch.tensor(labels, dtype=torch.float32)  # 多标签要用 float32

# ========== 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)

# ========== 数据编码 ==========
def encode_texts(texts, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
    return torch.stack(input_ids), torch.stack(attention_masks)

train_inputs, train_masks = encode_texts(X_train, tokenizer, max_len)
test_inputs, test_masks = encode_texts(X_test, tokenizer, max_len)

# 创建 DataLoader
train_dataset = TensorDataset(train_inputs, train_masks, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_inputs, test_masks, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========== 优化器和调度器 ==========
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 多标签损失函数（模型内部已包含 BCEWithLogitsLoss，这里无需手动定义）
# 但为了显式控制，也可以使用 nn.BCEWithLogitsLoss()

# ========== 训练循环 ==========
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, labels_batch = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_batch = labels_batch.to(device)  # shape: [batch, num_labels]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # 每个 epoch 结束后评估一次
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels_batch = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, num_labels]
            probs = torch.sigmoid(logits)  # 概率
            preds = (probs > 0.5).int()  # 阈值 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 多标签指标
    accuracy = accuracy_score(all_labels, all_preds)  # 严格匹配
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)

    print(f'Test Accuracy (exact match): {accuracy:.4f}')
    print(f'Samples Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    # 可选：按类别输出指标
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for i in range(num_labels):
        print(f'Class {i} ({label_names[i]}): P={per_class_precision[i]:.3f}, R={per_class_recall[i]:.3f}, F1={per_class_f1[i]:.3f}')

# ========== 保存模型 ==========
torch.save(model.state_dict(), 'bert_license_multilabel.pth')

# ========== 加载模型 ==========
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels, problem_type="multi_label_classification").to(device)
model.load_state_dict(torch.load('bert_license_multilabel.pth'))
model.eval()

# ========== 推理示例 ==========
def predict_licenses(text):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    predicted_labels = [label_names[i] for i, p in enumerate(probs) if p > 0.5]
    return predicted_labels

# 测试
sample = "MIT license and Apache License 2.0"
print(predict_licenses(sample))  # 输出 ['MIT', 'Apache-2.0']
```

处理后数据：

```json
{
 "raw":{
   "
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Assets: CC BY-SA 4.0](https://img.shields.io/badge/Assets-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
## License
OpenFront source code is licensed under the **GNU Affero General Public License v3.0**
Current copyright notices appear in:
- Footer: © OpenFront and Contributors
- Loading screen: © OpenFront and Contributors
Modified versions must preserve these notices in reasonably visible locations.
See the [LICENSE](LICENSE) for complete requirements.
For asset licensing, see [LICENSE-ASSETS](LICENSE-ASSETS).  
For license history, see [LICENSING.md](LICENSING.md).
"
},
"license_name":{
  "AGPL-3.0", "CC BY-SA 4.0"
}
}
```

## Qwen3-8b识别许可证关系

````markdown
# 任务：解析开源项目的许可证关系表达式

你是一个开源许可证分析专家。  
输入是一个 JSON 对象，包含：
- `raw`：项目的许可证描述文本（可能包含条款、徽章、说明等）
- `license_name`：一个集合（Set），列出项目中出现的所有许可证名称

你的任务是：根据 `raw` 中的描述，推断这些许可证之间的逻辑关系，并输出一个**关系表达式**。  
表达式只允许使用以下符号：
- `&` 表示 **AND**（必须同时遵守）
- `|` 表示 **OR**（可选择其中之一）
- 括号 `(` `)` 用于改变优先级

## 关系规则
- 如果 `raw` 中明确按不同部分（源代码、资产、文档等）分别授权，没有选择词 → 使用 `&`
- 如果 `raw` 中出现 `or`、`either ... or`、`at your option`、`choose` 等词 → 使用 `|`
- 如果同时出现多级组合（例如 "you can choose MIT or GPL, and for assets CC BY"），需要正确使用括号和优先级：`|` 优先级低于 `&`，等价于逻辑运算规则。
- 单一许可证时，直接输出许可证名（无需操作符）

## 输出格式
严格输出以下 JSON，不得包含任何其他文字：

```json
{
  "expression": "许可证关系表达式"
}
```

表达式中的许可证名称必须直接取自 `license_name` 中的字符串，保持原样。

## 示例

### 示例1（简单 AND）
输入：
```json
{
  "raw": "Source code under AGPL-3.0, assets under CC BY-SA 4.0.",
  "license_name": ["AGPL-3.0", "CC BY-SA 4.0"]
}
```
输出：
```json
{
  "expression": "AGPL-3.0 & CC BY-SA 4.0"
}
```

### 示例2（简单 OR）
输入：
```json
{
  "raw": "You may use this project under either MIT or Apache-2.0.",
  "license_name": ["MIT", "Apache-2.0"]
}
```
输出：
```json
{
  "expression": "MIT | Apache-2.0"
}
```

### 示例3（混合，带括号）
输入：
```json
{
  "raw": "You can choose GPL-3.0 or MIT for the code, and the documentation is CC BY-SA 4.0.",
  "license_name": ["GPL-3.0", "MIT", "CC BY-SA 4.0"]
}
```
输出：
```json
{
  "expression": "(GPL-3.0 | MIT) & CC BY-SA 4.0"
}
```

### 示例4（单一许可证）
输入：
```json
{
  "raw": "This project is licensed under BSD-3-Clause.",
  "license_name": ["BSD-3-Clause"]
}
```
输出：
```json
{
  "expression": "BSD-3-Clause"
}
```

### 示例5（复杂 OR 组合）
输入：
```json
{
  "raw": "You may distribute this software under either (MPL-2.0 or LGPL-3.0) and for assets either CC0 or CC BY 4.0.",
  "license_name": ["MPL-2.0", "LGPL-3.0", "CC0", "CC BY 4.0"]
}
```
输出：
```json
{
  "expression": "(MPL-2.0 | LGPL-3.0) & (CC0 | CC BY 4.0)"
}
```

## 现在，请分析以下输入：
````

