# CL-MoE 与 Octopus：UCIT 实验设置及当前结果

更新时间：2026-09-06

本文档记录本次 CL-MoE 迁移到 UCIT benchmark 后的实际复现设置，并与
Octopus 仓库公开设置和论文 Table 1 结果对照。除特别注明外，CL-MoE 数值来自
服务器实验 `octopus_matched_seed42`，Octopus 数值来自论文表格中的
`Octopus (ours)`（不带 `†`）行。

## 1. 当前结论

- 两边使用相同的 UCIT 六任务顺序、相同的测试集、相同的任务指标和相同的
  Stage 2 训练子集抽样。
- 当前 CL-MoE 最终阶段（`Last`）宏平均为 **49.96**，Octopus 论文报告为
  **71.01**，相差 **-21.05** 个百分点。
- 当前 CL-MoE 只完成了最终阶段的六任务评测。完整持续学习下三角矩阵及
  CL-MoE `Avg` 尚未补跑，因此不能把当前 `Last=49.96` 与 Octopus
  `Avg=71.08` 直接比较。
- 两项工作的训练暴露并不完全相同：Octopus 对每个任务先用完整训练集执行
  Stage 1，再用指定子集执行受约束的 Stage 2；当前 CL-MoE 每个任务只在该
  指定子集上训练一次。这是目前最重要的复现差异。（这个是两个方法的差异，不属于实验设置差别）

## 2. 统一的 benchmark 设置

### 2.1 任务顺序与数据量

| 顺序 | 任务 | CL-MoE 训练样本 | Octopus Stage 2 样本 | 测试样本 | 指标 |
|---:|---|---:|---:|---:|---|
| 1 | ImageNet-R | 9,600 | 9,600 | 3,000 | Accuracy |
| 2 | ArxivQA | 9,600 | 9,600 | 3,000 | Accuracy |
| 3 | VizWiz | 9,600 | 9,600 | 3,000 | Caption 七指标平均 |
| 4 | IconQA | 9,600 | 9,600 | 3,000 | Accuracy |
| 5 | CLEVR-Math | 16,000 | 16,000 | 3,000 | Accuracy |
| 6 | Flickr30k | 3,200 | 3,200 | 3,000 | Caption 七指标平均 |

相同点：

- 任务顺序完全相同。
- CL-MoE 使用 NumPy `RandomState(42)` 复现 Octopus `path.json#N` 和
  `data_seed=42` 的子集选择。
- 四个问答/分类任务都使用 `mathruler.grader.grade_answer`。
- VizWiz 和 Flickr30k 都使用 COCO Caption 的 BLEU-1/2/3/4、METEOR、
  ROUGE-L、CIDEr，任务分数为七项算术平均。

不同点：

- Octopus Stage 1 脚本读取每个任务的完整训练 JSON；Stage 2 才使用上表的
  `#N` 子集。当前 CL-MoE 没有对应的完整数据 Stage 1。（方法差异）
- CL-MoE 将 Octopus 指令转换为旧版 LLaVA `conversations` schema，但不复制
  图片。训练和评测的 `--image_folder` 直接指向
  `/home/data1/lyk/Octopus`；没有建立图片软链接。

## 3. 训练设置总表

| 项目 | 当前 CL-MoE 复现 | Octopus 官方 UCIT 设置 | 是否一致 |
|---|---|---|---|
| 任务顺序 | ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR-Math → Flickr30k | 相同 | 是 |
| 基础模型 | 本地 Vicuna-7B v1.5 + CLIP ViT-L/14-336 + LLaVA v1.5 projector | `llava-hf/llava-1.5-7b-hf` | 名义架构相同，加载格式/代码栈不同 |
| 精度 | BF16，TF32 开启 | BF16 | 基本相同 |
| 每任务 epoch | 1 | Stage 1 为 1，Stage 2 为 1 | 不同（Octopus 两阶段） |
| 当前任务训练数据 | 固定匹配子集 | Stage 1 完整集；Stage 2 固定子集 | 只对 Stage 2 一致 |
| GPU 数 | 2 × RTX 4090 | 官方脚本 4 GPU | 不同 |
| 每卡 batch size | 8 | 4 | 不同 |
| 梯度累积 | 2 | `16 / 4 = 4` | 不同 |
| 全局有效 batch | `8 × 2 × 2 = 32` | `4 × 4 × 4 = 64` | 不同 |
| 学习率 | `2e-4` | `1e-4` | 不同 |
| weight decay | 0 | 脚本未显式指定，使用 Swift 默认值 | 未严格对齐 |
| warmup ratio | 0.03 | 0.05 | 不同 |
| scheduler | cosine（训练参数）/ DeepSpeed 配置含 WarmupLR 接口 | Swift/HF 默认调度配置 | 未严格对齐 |
| 最大训练长度 | 4096 | 4096 | 是 |
| 梯度检查点 | 开启 | 脚本未显式开启 | 不同/待确认框架默认值 |
| 优化器 | AdamW | Swift/HF 默认优化器 | 名义接近，版本与实现未锁齐 |
| 分布式/显存策略 | DeepSpeed ZeRO-3，参数和优化器 CPU Offload | Swift DDP，脚本未配置 ZeRO Offload | 不同 |
| 保存策略 | `save_steps=50000`，实际主要保留任务边界最终模型 | 每 50 step，最多 12 个 checkpoint | 不同 |
| 任务内断点续跑 | 当前没有可靠的 ZeRO step checkpoint；支持任务边界续跑 | 有较密集的 step checkpoint | 不同 |
| 数据加载 worker | 4 | 4 | 是 |
| 随机种子/子集 | 子集 seed 42 | `data_seed=42` | 子集一致 |

### 3.1 LoRA 与持续学习模块

| 项目 | 当前 CL-MoE | Octopus |
|---|---|---|
| PEFT 形式 | 自定义 MoE-LoRA | 标准单 LoRA |
| LoRA rank | 总 `r=32` | `r=48` |
| LoRA alpha | 64 | 96 |
| scaling `alpha/r` | 2 | 2 |
| 目标模块 | 自动发现所有 Linear | `all-linear` |
| 专家数 | 4 | 不使用 MoE 专家 |
| 单专家内部 rank | 8（`32 / 4`） | 不适用 |
| 推理附加结构 | 每层保留专家和 instance router | 合并后为单 LoRA 更新，无额外路由 |
| 历史数据 replay | 无 | 无 |

LoRA rank 不必强行设成相同值。它是方法内部容量的一部分：CL-MoE 的 rank 32
被拆给 4 个 rank-8 专家，并额外训练 router；Octopus 使用一个 rank-48 LoRA。
若把二者 rank 强行设为相同数字，实际结构、参数使用方式和推理计算仍不相同。
公平对比应优先统一数据、任务顺序、基础模型族、训练预算和评测协议，同时分别
报告各方法原生超参数；若研究容量敏感性，再增加参数量匹配的消融实验。

## 4. 方法机制差异

### 4.1 CL-MoE

当前复现代码的核心路径是：

1. 在线性层中放置 4 个 LoRA 专家。
2. instance router 根据 token hidden state 对专家输出进行 softmax 加权。
3. 训练期间采样 router 使用次数，得到每个任务的专家排序。
4. 新任务先从上一任务正式模型初始化并训练，生成
   `Only_Pretrain_1.5_MOE_2/<task>` 中间模型。
5. `params.py` 根据相邻任务的专家排序，以 `alpha=0.7` 对上一正式模型和当前
   中间模型的 LoRA 权重进行选择性加权，生成新的任务边界正式模型。

因此 CL-MoE 主要通过专家分工、路由和参数动量式合并缓解干扰。它带有额外
router/专家推理计算，并依赖训练期间统计的专家使用信息。

### 4.2 Octopus

Octopus 是 History-Free Gradient Orthogonalization（HiFGO）方法：

1. Stage 1 在当前任务上做无约束 LoRA 适配，强调可塑性。
2. 对每个历史 adapter，使用当前任务分布中的 256 个样本估计历史参数梯度
   （GPWC），不保存或回放历史任务数据。
3. Stage 2 在当前任务子集上加入梯度正交/约束项，降低当前更新对历史知识的
   干扰；大多数任务使用 `lambda_1=0.02, lambda_2=0.01`，Flickr30k 使用
   `lambda_2=0.05`，首任务两个系数为 0。
4. 每个任务产生增量 LoRA；评测时按历史顺序逐个 merge 到基础模型。

因此 Octopus 主要在优化空间中控制梯度冲突，不扩展专家结构；合并完成后没有
MoE router 的推理开销，但训练流程包含完整数据 Stage 1、梯度估计和受约束
Stage 2，训练成本更高。

## 5. 评测设置

| 项目 | 当前 CL-MoE | Octopus | 对齐情况 |
|---|---|---|---|
| 测试集 | 每任务固定 3,000 条 | 每任务固定 3,000 条 | 是 |
| 解码 | Transformers `generate`，greedy，temperature 0 | vLLM `SamplingParams(temperature=0)` | 策略一致，实现不同 |
| 最大新生成 token | 16 | vLLM 0.7.3 UCIT 默认 16 | 是 |
| 最大上下文长度 | 4096 | 4096 | 是 |
| prompt 模板 | LLaVA `vicuna_v1` | 显式 LLaVA/Vicuna system prompt | 语义接近，序列化实现可能不同 |
| 推理并行 | 每 GPU 独立处理一个数据分片 | vLLM tensor parallel | 不同，但原则上不应改变 greedy 输出 |
| Accuracy grader | `mathruler.grade_answer` | 相同 | 是 |
| Caption scorer | `pycocoevalcap` | 相同 | 是 |

CL-MoE caption 评分复用了 Octopus 环境中的 `mathruler`、`pycocotools` 和
`pycocoevalcap`。Java 11 仅安装在 Octopus Conda 环境中，用于
PTBTokenizer、METEOR 和 SPICE 依赖，不是系统级安装。

## 6. 当前指标结果

### 6.1 CL-MoE 最终阶段（Last）

| 任务 | 指标 | CL-MoE |
|---|---|---:|
| ImageNet-R | Accuracy | 47.10 |
| ArxivQA | Accuracy | 74.47 |
| VizWiz | Caption 七项平均 | 43.01 |
| IconQA | Accuracy | 42.23 |
| CLEVR-Math | Accuracy | 38.53 |
| Flickr30k | Caption 七项平均 | 54.43 |
| **宏平均** | **Last** | **49.96** |

VizWiz 详细指标：BLEU-1 64.09、BLEU-2 47.75、BLEU-3 34.82、BLEU-4
24.88、METEOR 20.11、ROUGE-L 46.01、CIDEr 63.38。

Flickr30k 详细指标：BLEU-1 76.25、BLEU-2 59.97、BLEU-3 45.75、BLEU-4
34.50、METEOR 26.26、ROUGE-L 55.52、CIDEr 82.78。

### 6.2 与 Octopus 论文 Last 对比

| 方法 | ImageNet-R | ArxivQA | VizWiz | IconQA | CLEVR-Math | Flickr30k | Last |
|---|---:|---:|---:|---:|---:|---:|---:|
| Octopus (ours) | 88.20 | 93.03 | 58.46 | 60.50 | 69.00 | 56.87 | **71.01** |
| 当前 CL-MoE | 47.10 | 74.47 | 43.01 | 42.23 | 38.53 | 54.43 | **49.96** |
| **CL-MoE − Octopus** | **-41.10** | **-18.56** | **-15.45** | **-18.27** | **-30.47** | **-2.44** | **-21.05** |

差距集中在 ImageNet-R 和 CLEVR-Math；Flickr30k 最接近。当前 CL-MoE 的
`Last=49.96` 比论文 Sequential Finetune 的 `48.12` 高 1.84，但比论文
MoELoRA 的 `52.06` 低 2.10。

### 6.3 Octopus 论文 Avg 与 Last

| Octopus 版本 | Avg | Last | 说明 |
|---|---:|---:|---|
| Octopus (ours) | 71.08 | 71.01 | 主对比设置 |
| Octopus (ours)† | 71.33 | 70.45 | 使用 Historical task proxy approximation |

主对比应使用不带 `†` 的行。带 `†` 行包含额外近似设置，不应与当前 CL-MoE
默认配置混为同一组。

### 6.4 CL-MoE 训练统计

| 任务 | step | train loss | runtime (s) | samples/s |
|---|---:|---:|---:|---:|
| ImageNet-R | 300 | 0.2563 | 2,968.8 | 3.234 |
| ArxivQA | 300 | 0.1504 | 4,562.4 | 2.104 |
| VizWiz | 300 | 1.8292 | 3,680.2 | 2.609 |
| IconQA | 300 | 0.3830 | 3,729.9 | 2.574 |
| CLEVR-Math | 500 | 0.3513 | 5,689.9 | 2.812 |
| Flickr30k | 100 | 1.7336 | 1,248.9 | 2.562 |

不同任务的 loss 标签空间和回答形式不同，不能仅凭 loss 横向判断任务效果；最终
测试指标才是有效比较依据。

## 7. Avg、Last 与持续学习矩阵状态

- `Last`：第六任务训练完成后的模型在六个测试集上的分数及其宏平均。当前已
  完成，CL-MoE 为 **49.96**。
- `Avg`：对每个任务，从该任务刚学完的阶段一直到最终阶段取纵向平均，再对六
  个任务等权平均。当前 CL-MoE 尚缺前五个阶段的 15 个评测单元。
- 完整矩阵脚本：`scripts/UCIT/evaluate_matrix.sh`。
- 汇总脚本：`scripts/UCIT/summarize_matrix.py`。
- 计划输出：
  `results/UCIT/octopus_matched_seed42/matrix_summary.{json,md}`。

矩阵评测因服务器两张 GPU 当前被其他用户的 StableVLA 实验占用而暂缓。待资源
空闲后运行；脚本会识别并跳过已有的 6 个 Final 单元，只补跑缺失项。

## 8. 如何解释当前差距

现阶段不能把 -21.05 全部归因于持续学习算法优劣，至少混有以下变量：

1. **训练预算不同**：Octopus 每任务两阶段，CL-MoE 每任务一阶段。
2. **数据暴露不同**：只对齐了 Octopus Stage 2 子集，未复现其完整数据 Stage 1。
3. **有效 batch 不同**：32 对 64。
4. **学习率和 warmup 不同**：`2e-4/0.03` 对 `1e-4/0.05`。
5. **模型加载栈不同**：旧版 LLaVA/Vicuna 组件加载对 Hugging Face 一体化
   LLaVA checkpoint，无法保证权重和预处理逐字节一致。
6. **方法容量不同**：4 专家 rank-32 MoE-LoRA 对单 rank-48 LoRA。
7. **推理实现不同**：Transformers 分片推理对 vLLM tensor parallel；greedy
   设置虽已对齐，底层数值路径仍不同。

因此当前结果适合作为“按各自原生方法配置、统一 benchmark 外部标准”的
practical comparison，而不是严格只改变算法模块的 controlled ablation。

若要进一步收紧可比性，建议按优先级执行：

1. 完成 CL-MoE 持续学习矩阵，得到真正可对比的 `Avg`。
2. 对两个项目记录并校验基础模型权重、tokenizer、image processor 的哈希或
   配置差异。
3. 增加训练预算对齐实验：控制每任务看到的样本总数和 optimizer update 数。
4. 增加 global batch、学习率和 warmup 对齐实验。
5. 保留各自原生 LoRA 设置作为主结果，另做可训练参数量匹配的容量消融。
