---
icon: lightbulb
sidebar: false
date: 2025-04-08
prev: ./032_sft_trainer_sourcecode_prepare_model
next: ./030_wandb
category:
  - LLM
tag:
  - QLoRA
---
# QLoRA 代码实现及过程分析
- 背景介绍: QLoRA/基础模型/数据集
- QLoRA 代码实现
- QLoRA 过程分析
- QLoRA 应用价值
- QLoRA 疑点思考
- QLoRA 细节补充
<!-- more -->
## 1. 背景介绍: QLoRA/基础模型/数据集
- QLoRA
    - QLoRA（Quantized Low-Rank Adaptation）微调方法
    - Paper: https://arxiv.org/abs/2305.14314
    - QLoRA 结合了 4-bit 量化和 LoRA 技术，具体实现步骤如下：
        - 4-bit 量化：使用 bitsandbytes 库实现 4-bit NormalFloat (NF4) 量化，将预训练模型权重压缩至 4 位，显著降低内存占用
        - LoRA：通过 peft 库实现 LoRA，添加低秩适配器（例如秩 r=16），仅更新少量参数
        - 结合技术：加载量化后的模型，附加 LoRA 适配器，使用 16-bit (bfloat16) 进行前向/反向传播计算
- 基础模型：DistilGPT-2
    - https://huggingface.co/distilbert/distilgpt2
- Alpaca指令数据集
    - 官方版本
        - https://huggingface.co/datasets/tatsu-lab/alpaca
        - 使用 OpenAI 的 text-davinci-003 模型生成的输出，该数据集可能包含错误或偏见，建议用户谨慎使用并考虑过滤方法，这在数据集描述中有所提及
    - yahma/alpaca-cleaned 
        - https://huggingface.co/datasets/yahma/alpaca-cleaned
        - 清理版本，修复了原始数据集中的幻觉、错误答案等问题，提供了高质量的数据
    - vicgalle/alpaca-gpt4
        - https://huggingface.co/datasets/vicgalle/alpaca-gpt4
        - 基于 Alpaca 提示，使用 GPT-4 生成的输出，，但未提及清理，可能包含未修正的错误

## 2. QLoRA 代码实现
- Load Model
- Preparing Dataset
- Fine-Tuning
- Save Trained Model

### 2.1. Load Model
```python
# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 启用 4bit 量化，将模型的线性层（Linear / Conv1D）替换成量化层 Linear4bit
    bnb_4bit_use_double_quant=True, # 启用嵌套量化，进一步压缩量化参数，减少存储开销 (Linear4bit内部计算逻辑)
    bnb_4bit_quant_type="nf4", # 4bit 量化格式有2种（nf4和fp4），其中nf4基于正态分布优化，通常效果更优
    bnb_4bit_compute_dtype=torch.bfloat16 # 设置计算时的数据类型，实际权重以 4bit 存储但会映射到 bfloat16 进行计算，也就是 Linear4bit 内部的中间计算使用 bfloat16
)
```

```python
# 选择 distilbert/distilgpt2 作为基础模型
model_id = "distilbert/distilgpt2"

# 将整个模型加载到 GPU 0
device_map = {"": 0}

# 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained(model_id)

# 加载量化模型（将量化配置应用在模型上）
quantized_model = AutoModelForCausalLM.from_pretrained(model_id,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    use_cache = False)
```

```python
# 加载与模型对应的分词器，并设置填充标记为结束标记
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
```

### 2.2. Preparing Dataset
```python
# 选择 yahma/alpaca-cleaned 作为数据集
dataset_name = "yahma/alpaca-cleaned"

# 加载数据集
full_dataset = load_dataset(dataset_name, split="train")

# 选取小规模子集（1000 条）
small_subset = full_dataset.shuffle(seed=42).select(range(1000))
```

```python
# 定义 Alpaca 数据集的 Prompt 模版
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 定义 formatting_prompts_func 函数
def formatting_prompts_func(examples):

    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]

    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output)
        texts.append(text)
    return { "text" : texts }


# 应用 formatting_prompts_func 函数
small_subset = small_subset.map(formatting_prompts_func, batched=True,)
```

```python
# 对 "text" 应用 tokenizer；如果超长，截断到模型最大长度；所有样本 pad 到相同长度，方便 batch 训练
small_subset = small_subset.map(lambda samples: tokenizer(samples["text"], truncation=True, padding="max_length"), batched=True)
```

### 2.3. Fine-Tuning
```python
# LoRA 参数配置
peft_config = LoraConfig(
    r=8, # 秩，越大表达能力越强，但参数也更多
    lora_alpha=16, # 缩放因子
    lora_dropout=0.05, # dropout 概率
    target_modules=["c_attn", "c_proj", "c_fc"],  # 需要插入 LoRA 的模块
    bias="none", # 是否训练 bias 项：否
    task_type="CAUSAL_LM", # 任务类型：因果语言建模
)

# 训练参数配置
training_args = SFTConfig(
    output_dir="outputs", # 输出路径
    logging_steps=1, # 多少steps记录一次日志
    num_train_epochs=3, # 训练轮数
    per_device_train_batch_size=2, # 每个设备的训练批次大小
    per_device_eval_batch_size=2, # 每个设备的验证批次大小
    gradient_accumulation_steps=5, # 梯度累积
    gradient_checkpointing=True, # 启用梯度检查点
    learning_rate=2e-4, # 学习率
    optim="adamw_8bit", # 优化器
    weight_decay=0.01, # 权重衰减
    max_grad_norm=0.3, # 梯度裁剪
    warmup_ratio=0.03, # 预热比例
    fp16=not torch.cuda.is_bf16_supported(), # 使用半精度训练
    bf16=torch.cuda.is_bf16_supported(),
    dataset_text_field="text",
)

# 实例化 SFTTrainer
trainer = SFTTrainer(
    model=quantized_model,
    train_dataset=small_subset,
    peft_config=peft_config,
    args=training_args,
)

# 输出可训练参数量
trainer.model.print_trainable_parameters()

# 开始训练
trainer.train()
```

### 2.4. Save Trained Model
```python
# Save trained model
peft_model = "distilgpt2-qlora"

trainer.model.save_pretrained(peft_model)
```

```python
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

peft_model = PeftModel.from_pretrained(base_model, peft_model)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")
```

## 3. QLoRA 过程分析
- 原始模型和量化模型的对比
- Dataset 处理流程
- 可训练参数量计算

### 3.1. 原始模型和量化模型的对比
- 参数数量不会变（还是那么多矩阵元素）
  - 81912576
- 参数精度和大小变了（用 4-bit 表示）
  - 参数大小变化
    - Original size: 318.47 MB
      - 估算：81912576 * 4 bytes / (1024^2) = 308.66 MB
    - Quantized size: 101.49 MB
      - 估算
        - 0.5 bytes 的参数个数：42467328 = 6 layers * (768 * 2304 + 768 * 768 + 2 * 768 * 3072)
        - 2 bytes 的参数个数：39445248 = 81912576 - 42467328
        - （0.5 * 42467328 + 2 * 39445248 ） / (1024^2) = 95.49 MB
  - 模型结构变化
    - attn
      - (c_attn): Conv1D(nf=2304, nx=768) -> (c_attn): Linear4bit(in_features=768, out_features=2304, bias=True)
      - (c_proj): Conv1D(nf=768, nx=768) -> (c_proj): Linear4bit(in_features=768, out_features=768, bias=True)
    - mlp
      - (c_fc): Conv1D(nf=3072, nx=768) -> (c_fc): Linear4bit(in_features=768, out_features=3072, bias=True)
      - (c_proj): Conv1D(nf=768, nx=3072) -> (c_proj): Linear4bit(in_features=3072, out_features=768, bias=True)
  - 参数精度变化
    - 量化前：
      - 所有参数全都是dtype=torch.float32（32位）
        - transformer.h.0.attn.c_attn.weight: torch.Size([768, 2304]), dtype=torch.float32
        - transformer.h.0.attn.c_proj.weight: torch.Size([768, 768]), dtype=torch.float32
        - transformer.h.0.mlp.c_fc.weight: torch.Size([768, 3072]), dtype=torch.float32
        - transformer.h.0.mlp.c_proj.weight: torch.Size([3072, 768]), dtype=torch.float32
    - 量化后：
      - dtype=torch.uint8（每层的4个地方变为4bit）实际存储中使用压缩技术，将2个4bit组合为int8
        - transformer.h.0.attn.c_attn.weight: torch.Size([884736, 1]), dtype=torch.uint8
        - transformer.h.0.attn.c_proj.weight: torch.Size([294912, 1]), dtype=torch.uint8
        - transformer.h.0.mlp.c_fc.weight: torch.Size([1179648, 1]), dtype=torch.uint8
        - transformer.h.0.mlp.c_proj.weight: torch.Size([1179648, 1]), dtype=torch.uint8
      - dtype=torch.float16（其余参数都变为float16）


变化解释
- 变化
  - 量化前：transformer.h.0.attn.c_attn.weight: torch.Size([768, 2304]), dtype=torch.float32
  - 量化后：transformer.h.0.attn.c_attn.weight: torch.Size([884736, 1]), dtype=torch.uint8
- 解释
  - 原始 float32 的矩阵 [768, 2304] → 总共参数数量是 768 * 2304 = 1,769,472
  - 用 4-bit 表示就是 1,769,472 * 0.5 byte = 884,736 bytes = 884736 uint8 （存储为 packed 的 uint8，每 byte 存两个 4-bit 权重）
  - [884736, 1]正是把原来的权重展开成1维后量化存储的结果

### 3.2. Dataset 处理流程
- 1.加载数据集
    - 数据集有3列
        - instruction
        - input
        - output
    - 数据集有 51760 条数据
```
Dataset({
    features: ['output', 'input', 'instruction'],
    num_rows: 51760
})
```
- 2.选取小规模子集
    - 随机选取 1000 条数据
```
Dataset({
    features: ['output', 'input', 'instruction'],
    num_rows: 1000
})
```
- 3.拼接 'output', 'input', 'instruction' 这3个字段为一个字符串，作为 'text' 字段
```
Dataset({
    features: ['output', 'input', 'instruction', 'text'],
    num_rows: 1000
})
```
- 4.将 "text" 字段采用 tokenizer 进行 token 化，生成 "input_ids" 和 "attention_mask" 字段
```
Dataset({
    features: ['output', 'input', 'instruction', 'text', 'input_ids', 'attention_mask'],
    num_rows: 1000
})
```

### 3.3. 可训练参数量计算
- 基础模型参数量
  - 81912576
- 模型结构中的目标模块
  - attn
    - (c_attn): Conv1D(nf=2304, nx=768) -> (c_attn): Linear4bit(in_features=768, out_features=2304, bias=True)
    - (c_proj): Conv1D(nf=768, nx=768) -> (c_proj): Linear4bit(in_features=768, out_features=768, bias=True)
  - mlp
    - (c_fc): Conv1D(nf=3072, nx=768) -> (c_fc): Linear4bit(in_features=768, out_features=3072, bias=True)
    - (c_proj): Conv1D(nf=768, nx=3072) -> (c_proj): Linear4bit(in_features=3072, out_features=768, bias=True)
- 参数之间的关系
  - 总参数量 82,502,400 - QLoRA可训练参数量 589,824 = 基础模型参数量 81912576
  - QLoRA可训练参数量 589,824 = 6层 * (24576 + 12288 + 30720 + 30720) = 6 * 98304
    - 共6层，每层4个目标模块
      - c_attn: 768 * 8 + 2304 * 8 = 24576
      - c_proj: 768 * 8 + 768 * 8 = 12288
      - c_fc: 3072 * 8 + 768 * 8 = 30720
      - c_proj: 768 * 8 + 3072 * 8 = 30720

## 4. QLoRA 应用价值
### 4.1. QLoRA 和全参数微调⽅法对比

- 微调参数量：  
  - QLoRA：仅微调一小部分参数（LoRA Adapter），在本例中，可训练参数仅占总参数的 0.7149%
  - 全参数微调：更新所有模型参数，需要更多内存和计算资源
- 内存使用：  
  - QLoRA：使用 4-bit 量化，内存占用大幅减少
  - 全参数微调：需要全精度（例如 16 位或 32 位），通常需要较大的 GPU 内存
- 训练速度：  
  - QLoRA：由于参数少且精度低，训练更快  
  - 全参数微调：更新所有权重，训练时间较长
- 性能：  
  - QLoRA：由于量化和有限的参数更新，性能可能略低于全参数微调，但仍保留大部分能力
  - 全参数微调：因所有权重都被优化，性能可能更高
- 适用场景：  
  - QLoRA：适合资源受限环境或快速实验  
  - 全参数微调：适合资源充足且需最大精度的场景

### 4.2. QLoRA 在该任务中的优势和潜在局限性

优势：
- 内存效率：4 位量化和 LoRA 减少了内存使用，使微调能在较小的 GPU 上运行（例如 Colab 免费版）
- 速度：由于参数少且精度低，训练速度更快，适合快速迭代
- 保留预训练知识：冻结大部分权重保留了基础模型的泛化能力，同时适配任务
- 适用于小数据集：对像 1,000 条 Alpaca 子集这样的小数据集效果良好，不易过拟合

局限性：
- 性能权衡：量化和 LoRA 可能导致性能略低于全参数微调，尤其在复杂任务中  
- 任务特定性：LoRA 适配器是任务特定的，切换任务需重新训练或维护多个适配器  
- 量化噪声：4-bit 精度引入噪声，可能影响输出质量  
- 超参数敏感性：调整 `r`、`lora_alpha` 和量化设置需要实验，增加了复杂性

## 5. QLoRA 疑点思考
### 5.1. lora和qlora微调的方式是不是只有训练前的模型是否进行了量化这一区别？

是的，主要的区别就是训练前的模型是否已经进行了量化
- LoRA 微调的是一个 未量化的预训练模型，权重通常是 float32。
- QLoRA 微调的是一个 已量化的预训练模型，例如 4bit 或 8bit。权重是 量化后的 uint8 或 int8 格式。

内存与推理效率：
- LoRA 在微调时占用的内存较大，因为它使用的是 全精度 的预训练模型（通常是 float32）。
- QLoRA 的优势在于 内存占用大大降低，因为模型已经量化到 4bit。即使是大模型，也能在有限的显存中运行。

相同点
- LoRA 和 QLoRA 都是通过训练一个 低秩适配矩阵 来微调原始模型，无需修改原始的预训练权重
- 两者都只训练适配层，因此参数量相对较少，适合资源有限的场景

LoRA（Low-Rank Adaptation）微调步骤
- 加载一个 预训练模型（通常是 float32 权重）。
- 冻结原始权重，只训练 LoRA 的适配矩阵。
- 微调时，通过训练 LoRA 层来使模型适应特定任务。
- 推理时，使用 原始权重 加上训练好的 LoRA 层的适配矩阵

QLoRA（Quantized LoRA）微调步骤
- 量化预训练模型：加载一个 已经量化的模型，例如 4bit 或 8bit 模型。
- 冻结量化后的权重，只训练 LoRA 适配矩阵。
- 在 量化后的模型 上进行微调，优化 LoRA 层的适配矩阵。
- 推理时，仍然使用 量化后的权重，加上训练好的 LoRA 适配矩阵。

### 5.2. 为什么量化通常只用于线性层？
原因可以从以下几个角度来解释：数学、工程实现、对模型影响

- 1.线性层是参数最多、计算量最大的部分（参数集中）
  - 层归一化（LayerNorm）、激活函数（GELU）、Dropout 等几乎不含权重或只有极少量参数
  - 所以优先量化线性层，收益最大，且对模型结构改动最小
- 2.线性层结构简单、便于量化和解码（数学结构简单）
  - 线性层是少数可以同时量化权重 + 激活 + 梯度的层（如果需要），所以它成为了主战场
  - 量化通常分为两种
    - 权重量化（Weight Quantization）：将 weight 低精度存储（最常见）
    - 激活量化（Activation Quantization）：对中间值进行量化（更复杂）
  - 线性变换是一个非常清晰、固定的数学结构，适合被：
    - 编码为 4-bit / 8-bit 数组
    - 解码时用 scale + zero_point 进行恢复
    - 用 kernel-fusion 加速（比如 bitsandbytes 提供的 CUDA 实现）
- 3.非线性操作不好量化（其他层不适合，量化线性层影响较小）
  - 这些层要么不含参数，要么行为变化大，量化这些部分带来的收益不明显，反而增加复杂度和误差
    - 非线性函数（GELU, ReLU）没有固定权重，不好提前编码
    - 归一化（LayerNorm, BatchNorm）涉及动态计算均值、方差、除法，不好静态量化
    - 控制结构（Dropout, Mask）行为在训练/推理不同，量化意义不大
- 4.工程上已经高度优化了线性层量化（工程已优化）
  - 现有框架都专注优化了 Linear, Conv 这些算子，对非线性操作支持很差甚至没有
    - bitsandbytes
    - Intel Neural Compressor
    - NVIDIA TensorRT
    - ONNX Runtime + QNNPACK

### 5.3. 为什么不对 lm_head 层进行量化，这个也是线性层?

按道理讲，这个层也可以量化，但它默认并没有被量化

原因主要是以下几个：

- 1.lm_head 用于输出 → 精度影响更敏感 （精度敏感）
  - lm_head 是模型的最后一层，用于将隐藏状态投影到词表（vocab）空间，输出 logits
  - 这一层的输出直接影响：
    - softmax 的分布
    - 预测的 token 排序
    - 最终生成文本的质量
  - 所以，它对精度非常敏感。轻微的权重误差可能就会导致 token 排序错误，生成完全不同的结果
- 2.大多数量化框架默认跳过 lm_head （框架默认行为，空间节省有限）
  - 比如 bitsandbytes、AutoGPTQ 等，在量化 Transformer 模型时默认不对 lm_head 做 4-bit 量化，因为：
    - 这一层通常只有一份（不像 attention 和 MLP 有很多层）
    - 精度影响更大
    - 节省空间有限（相对整模型来说），量化意义不大
- 3.可能用于权重共享（tie_weights）（权重共享，与 embedding 权重绑定，量化可能不兼容）
  - 有些模型中，lm_head.weight 是和嵌入层 wte.weight 共享权重的（tie-weights）
  - 此时量化 lm_head 就会影响嵌入 → 编码 → 解码的整个闭环
  - 如果你量化了 lm_head，但没同步量化 wte，或者反之，可能导致不一致甚至报错

### 5.4. 为什么非线性层在量化后的dtype也有变化，变为float16？

虽然没有显式设置这些层的 dtype，但它们会在 load_in_4bit=True 时自动被转换成更轻量的精度

这其实是 transformers 的 AutoPrecision 推理机制（自动调度）的一部分，和 bitsandbytes 一起配合使用

### 5.5. 量化模型精度发生了什么变化?

- 原始模型精度为 float32 (磁盘) 
- 量化模型的权重以 int4 存储（packed uint8） （显存）
- input float32/bfloat16 (外部传入)
- 权重 int4 临时解码成 bfloat16 计算
- double quant → 再压缩 scale，减少负担
- output bfloat16 → 后续层可以继续低精度执行

Note1：只有线性层的存储为4bit，其他层的存储、计算、解码输出这些都是bfloat16/float16

Note2：推理时不会改变模型的参数，只有反向传播时才会改变，推理时bit4的临时变为bfloat16进行计算，然后输出结果output也是bfloat16类型的

Note3：微调量化模型（比如 LoRA + 4bit），也是只更新部分 float 参数（如 LoRA adapter），4bit 权重本身不会直接修改

### 5.6. bnb_4bit_compute_dtype=torch.bfloat16是做什么的，计算精度和存储精度不同？
默认情况，即使你量化了参数，计算时还是 float32

所以可以设置计算精度为torch.bfloat16 或 torch.float16来加速计算过程，减少显存占用

计算精度这个 dtype 是在 Linear4bit 内部运行时设置的，属于内部计算逻辑

### 5.7. 临时性量化与永久存储？
- 临时性量化
  - 加载原始的 float32 权重模型，根据量化配置进行量化存在内存中
  - 推理时从内存中获取4bit临时转为bfloat16进行计算
  - 关键
    - 量化模型的 4bit 权重只在内存中存在，它们是临时的，不会被永久保存
    - 并没有改变磁盘上的原始模型文件，而是只影响了内存中的计算方式
- 永久保存
  - 保存模型的量化版本，加载模型时直接加载量化后的模型，不需要进行量化配置

### 5.8. qlora存储训练好的模型是按什么精度存储的？
QLoRA 在训练过程中的量化和存储过程有两种常见的策略
- 情况 1：训练时量化，存储时保留原始模型的 float32 权重
  - 原始模型的权重仍然会恢复为 float32 格式，而不是以量化后的格式保存
  - 量化操作通常是在推理时临时应用的，量化权重在训练时只是为了节省内存和加速计算，但不一定要永久存储
  - 将原始模型保存在 float32 格式 可以确保更好的模型兼容性和后续的易用性
  - LoRA 适配层：LoRA 层的适配权重 仍然是 float32 格式，会和原始模型一起存储
  - QLoRA 保存的模型是：
    - 存储的是原始模型的权重float32格式。
    - 存储的是 LoRA 适配层的权重，这些适配层依然是 float32 精度的。
- 情况 2：训练和存储时都量化
  - 这种方法主要用于 节省存储空间 和 加速推理
  - QLoRA 保存的模型是：
    - 存储的是 量化后的模型，即 4bit（通常是 uint8）权重。
    - 存储的是 LoRA 适配层的权重，这些适配层依然是 float32 精度的。

不需要将 LoRA 适配层权重和模型权重的数据类型保持一致吗？答：数据类型不必完全一致，原因如下
- LoRA 适配层与原始模型权重的独立性
- LoRA 层权重和量化后的模型权重的数据类型不一致并不冲突
  - 推理时，量化的权重（例如 4bit）会通过反量化过程恢复为 float16 或 bfloat16 格式进行计算，而 LoRA 层的权重（float32）依然参与计算，二者的精度不一致不会导致问题

## 6. QLoRA 细节补充
### 6.1. device_map
device_map 是一个字典，用于告诉模型加载器（比如 transformers 中的 AutoModel.from_pretrained）将模型的各个部分加载到哪些设备上。键（key）通常表示模型的某个部分（例如层或模块的名称），值（value）表示设备编号或设备名称（例如 GPU 的索引号 0、1，或者 "cpu"）。

```
device_map={"": 0}  # 将整个模型加载到 GPU 0（设备编号为 0 的设备上）（空字符串表示整个模型；在 PyTorch 中，0 通常对应于第一个 GPU（即 cuda:0））
```
```
device_map = {
    "transformer.layer.0": 0,  # 第 0 层加载到 GPU 0
    "transformer.layer.1": 1   # 第 1 层加载到 GPU 1
}
```
```
device_map = "auto" # 根据可用设备自动分配模型（需要 accelerate 库支持）
```

延伸：如何检查 GPU 可用设备
```
import torch
print(torch.cuda.is_available())  # 检查是否有 GPU
print(torch.cuda.device_count())  # 检查 GPU 数量
print(torch.cuda.current_device())  # 当前默认 GPU 编号
``` 

### 6.2. seed=42 是常见“魔法数字”
42 是个惯例值，出自《银河系漫游指南》，表示“生命、宇宙以及一切问题的终极答案”。当然你用 123 或别的值也完全没问题，只要保证每次用一样的 seed就能复现

## 7. Reference
- huggingface SFTTrainer: https://huggingface.co/docs/trl/v0.7.4/en/sft_trainer
- google: https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
- unsloth: https://huggingface.co/blog/Andyrasika/finetune-unsloth-qlora