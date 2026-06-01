# CL-MoE 复现验证工作总结

## 背景

在复现 CL-MoE 的过程中，当前评测结果明显偏高：

- 经过 CL-MoE LoRA 微调后的结果高于论文报告值。
- 未加载 LoRA 的 BaseModel 评测结果甚至高于微调模型。
- 部分 BaseModel 单项结果接近或超过论文中 Multitask upper bound 的结果。

因此我们主要验证的问题是：BaseModel 的高分到底来自评测代码问题、数据泄漏、答案分布偏置，还是来自 LLaVA 风格基座模型本身的视觉理解能力。

## 模型加载方式确认

当前训练脚本并不是直接加载完整的 `llava-v1.5-7b` MLLM checkpoint 作为基座模型。

实际组成是：

```text
Vicuna-7B-v1.5 LLM
+ CLIP ViT-L/14-336 视觉编码器
+ LLaVA-1.5 pretrained mm_projector
+ continual training 阶段插入 CL-MoE LoRA
```

相关训练参数是：

```bash
--model_name_or_path checkpoint/vicuna-7b-v1.5
--vision_tower checkpoint/clip-vit-large-patch14-336
--pretrain_mm_mlp_adapter checkpoint/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin
```

因此，这个项目使用的是 LLaVA 架构和 LLaVA 预训练好的多模态 projector，但语言模型路径是 Vicuna。名为 `llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` 的目录实际只包含 projector/config 文件，不是完整的 LLaVA MLLM 权重。

## 远程权重目录检查

远程服务器上的相关目录情况如下：

- `checkpoint/vicuna-7b-v1.5`：完整 Vicuna 权重，约 13 GB。
- `checkpoint/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5`：只有 `config.json` 和 `mm_projector.bin`，约 41 MB。
- `checkpoints/CL4VQA/causal/llava-1.5-7b-lora`：包含 adapter/config/non-LoRA trainables，约 202 MB。

这基本排除了 BaseModel 评测时误加载完整微调版 LLaVA checkpoint 的可能。

## 新增诊断工具

为了排查问题，我们新增并推送了两类诊断工具。

1. `scripts/CLMoE/Eval/diagnose_clmoe_eval.py`

   这个脚本会输出：

   - 每个任务的 train/test 样本数
   - train/test 的 `question_id` 是否重叠
   - 测试集 majority-answer baseline
   - 任意 result stage 的逐任务准确率
   - 十个 CL4VQA 任务的平均 AP

2. `llava/eval/CLMoE/model_vqa_cc_instruction.py` 中新增图片扰动评测模式

   当前 eval 支持三种图片输入模式：

   - `normal`：原始评测，问题和图片正常对应。
   - `shuffle`：问题文本、`question_id`、标准答案保持不变，但测试图片随机错配。
   - `blank`：问题文本、`question_id`、标准答案保持不变，但图片统一替换成纯黑空白图。

   使用示例：

   ```bash
   bash scripts/CLMoE/Eval/Eval_all.sh base shuffle
   bash scripts/CLMoE/Eval/Eval_all.sh base blank
   python3 scripts/CLMoE/Eval/diagnose_clmoe_eval.py --stage BaseModel_shuffle
   python3 scripts/CLMoE/Eval/diagnose_clmoe_eval.py --stage BaseModel_blank
   ```

## 数据泄漏检查

诊断脚本显示，十个任务的 train/test `question_id` overlap 全部为 0：

```text
recognition  qid_overlap=0
location     qid_overlap=0
judge        qid_overlap=0
commonsense  qid_overlap=0
count        qid_overlap=0
action       qid_overlap=0
color        qid_overlap=0
type         qid_overlap=0
subcategory  qid_overlap=0
causal       qid_overlap=0
```

这说明不存在简单的 train/test 问题 ID 重复泄漏。

测试集 majority-answer baseline 也不足以解释 BaseModel 的正常高分。例如：

```text
recognition  majority='RIGHT': 1.05%
location     majority='BATHROOM': 4.70%
color        majority='WHITE': 18.44%
type         majority='NONE': 1.65%
causal       majority='NO': 3.69%
```

部分 yes/no 或二分类倾向较强的任务确实有较高 majority baseline：

```text
judge        majority='NO': 46.51%
commonsense  majority='YES': 53.38%
action       majority='NO': 32.04%
```

这些可以解释图片扰动后剩余的一部分分数，但无法解释正常图片输入下 BaseModel 的整体高分。

## BaseModel 正常评测结果

BaseModel 在正常图片输入下的结果为：

```text
recognition   55.87
location      43.02
judge         81.89
commonsense   78.16
count         52.81
action        77.62
color         74.75
type          61.60
subcategory   61.16
causal        28.11
AP(mean)      61.50
```

使用缩减训练数据后得到的 Finetune 结果为：

```text
recognition   55.08
location      39.32
judge         79.61
commonsense   77.10
count         50.88
action        75.55
color         74.01
type          61.19
subcategory   60.66
causal        26.27
AP(mean)      59.97
```

因为当前复现为了节省成本，训练样本量约为原始设置的十分之一，所以 Finetune 结果低于 BaseModel 是合理的。这一现象本身不是主要异常点。

真正需要解释的是：BaseModel 本身为什么已经高于论文中的部分结果。

## 不同样本量训练结果对比

为了验证“训练数据会影响/削弱原本较强的 BaseModel 能力”这一判断，我们在保持当前训练超参不变的前提下，比较了不同训练样本量下的 CL-MoE continual tuning 结果。

当前固定的训练超参为：

```bash
--expert_num 4
--lora_r 32
--lora_alpha 64
```

目前已有三组结果：

```text
BaseModel zero-shot AP = 61.50
1/10 Finetune AP      = 59.97
1/5 Finetune AP       = 60.09
```

逐任务对比如下：

| Task | BaseModel | 1/10 Finetune | 1/5 Finetune | 1/5 - 1/10 | 1/5 - Base |
| --- | ---: | ---: | ---: | ---: | ---: |
| recognition | 55.87 | 55.08 | 54.64 | -0.44 | -1.23 |
| location | 43.02 | 39.32 | 38.75 | -0.57 | -4.27 |
| judge | 81.89 | 79.61 | 80.04 | +0.43 | -1.85 |
| commonsense | 78.16 | 77.10 | 76.85 | -0.25 | -1.31 |
| count | 52.81 | 50.88 | 51.29 | +0.41 | -1.52 |
| action | 77.62 | 75.55 | 75.55 | +0.00 | -2.07 |
| color | 74.75 | 74.01 | 73.24 | -0.77 | -1.51 |
| type | 61.60 | 61.19 | 62.01 | +0.82 | +0.41 |
| subcategory | 61.16 | 60.66 | 61.78 | +1.12 | +0.62 |
| causal | 28.11 | 26.27 | 26.73 | +0.46 | -1.38 |
| **AP(mean)** | **61.50** | **59.97** | **60.09** | **+0.12** | **-1.41** |

从 1/10 增加到 1/5 后，平均 AP 只提升了 `+0.12`，提升并不明显。部分后序任务如 `type`、`subcategory`、`causal` 有一定提升，但 `recognition`、`location`、`color` 等任务反而下降。

这说明当前现象不能简单归因于“1/10 样本量太少”。更准确的判断是：

- BaseModel 本身已经很强。
- 当前 CL-MoE LoRA continual tuning 在多数任务上仍会削弱 BaseModel 的原始能力。
- 增加到 1/5 样本量后，削弱现象有所缓解但并未消失。
- CL-MoE 方法是否有效，应该继续与“不使用 CL-MoE 的普通 continual learning baseline”比较，而不是只和 BaseModel zero-shot 结果比较。

下一步实验将继续保持当前训练超参不变，只把样本量增加到约 1/2，以形成更干净的样本量曲线：

```text
BaseModel zero-shot
1/10 CL-MoE
1/5 CL-MoE
1/2 CL-MoE
```

1/2 实验对应的 checkpoint 目录规划为：

```text
checkpoints/CL4VQA_1_2/
checkpoints/CL4VQA_1_2/Only_Pretrain_1.5_MOE_1_2/
```

这样可以避免覆盖 1/10 和 1/5 已有结果。

## Shuffle 图片评测

`shuffle` 模式下，每个问题保留原始问题文本、`question_id` 和标准答案，但输入图片被随机替换为测试集中的另一张图片。

结果如下：

```text
recognition    9.43
location       3.56
judge         51.20
commonsense   50.04
count         11.84
action        36.88
color         16.28
type           6.85
subcategory   23.44
causal         4.61
AP(mean)      21.41
```

这说明 BaseModel 的正常高分强依赖正确图片输入。

如果高分主要来自问题文本泄漏或答案分布偏置，那么 shuffle 图片后不应该出现如此大幅下降。

## Blank 图片评测

`blank` 模式下，每个问题保留原始问题文本、`question_id` 和标准答案，但输入图片统一替换成纯黑空白图。

结果如下：

```text
recognition   10.14
location       3.28
judge         51.03
commonsense   51.02
count          9.50
action        35.50
color         16.61
type           8.67
subcategory   23.57
causal         6.45
AP(mean)      21.58
```

三种设置的平均 AP 对比：

```text
BaseModel normal   AP = 61.50
BaseModel shuffle  AP = 21.41
BaseModel blank    AP = 21.58
```

`blank` 和 `shuffle` 结果非常接近，说明分数下降主要来自有效视觉信息被移除，而不是某张错误图片带来了特殊干扰。

## 当前结论

现有证据支持以下结论：

1. BaseModel 的高分不能用简单的 train/test `question_id` 泄漏解释。

2. BaseModel 的高分不能只用 majority-answer bias 解释。

3. BaseModel 的高分不太可能来自 prompt 文本泄漏。若问题文本本身泄漏答案，`blank` 模式下分数仍应保持较高。

4. BaseModel 的高分主要依赖正确视觉输入。证据是正常评测 AP 为 `61.50`，而 `shuffle` 和 `blank` 都下降到约 `21.5`。

5. 图片扰动后剩余的约 `21.5` AP 主要来自语言先验和答案分布。这个现象在 `judge`、`commonsense`、`action` 等 yes/no 倾向较强的任务中尤其明显。

6. 论文中的 `Multitask` 结果不应理解为所有可能 base-model evaluation 的理论上界。它更像是在论文训练协议下的 multitask adapter/continual-learning setting 上界，不一定高于一个强 LLaVA 风格基座模型在同一测试集上的直接评测结果。

## 仍待确认的问题

目前主要未解决的问题是：为什么当前 LLaVA 风格 BaseModel 在多个任务上强于论文报告的 Multitask 结果。

可能原因包括：

- 论文没有报告当前这种 zero-shot/base evaluation 设置。
- 论文中的 `Multitask` 行是训练后的 adapter/continual-learning baseline，而不是完整 LLaVA base model 的直接评测。
- 作者实际使用的 LLaVA-1.5 projector、Vicuna checkpoint 或 CL4VQA split 与当前环境存在细节差异。
- 论文评测中的答案归一化或评分协议与当前 exact-match evaluator 不完全一致。
- 在 CL4VQA 上进行 LoRA/adapter 训练可能会损害部分 LLaVA 原有视觉语言能力，尤其是在当前只使用缩减训练数据的情况下。

## 后续建议

1. 在实验记录中保留以下三行诊断结果：

   ```text
   BaseModel normal   AP = 61.50
   BaseModel shuffle  AP = 21.41
   BaseModel blank    AP = 21.58
   ```

2. 如果作者提供了原始 evaluator，应对比当前 `eval_vqav2.py` 的 exact-match 评测协议与论文实际协议是否一致。

3. 继续确认 projector 来源。预期应为 LLaVA-1.5 pretrained projector，而不是从完整 instruction-tuned LLaVA 模型中抽取出的 projector。

4. 如果目标是严格复现论文最终 CL-MoE 数值，应恢复论文级训练超参：

   ```bash
   --expert_num 8
   --lora_r 64
   --lora_alpha 128
   ```

   同时需要移除或重新设置当前为了节省训练量而缩小的 `max_steps`。

5. 缩减数据训练的结果应和完整复现实验分开记录。当前缩减数据实验适合用于调试流程和验证现象，但不应期待与论文数值严格一致。
