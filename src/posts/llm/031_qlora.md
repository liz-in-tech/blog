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
# QLoRA Code Implementation and Process Analysis
- Background Introduction: QLoRA / Base Model / Dataset
- QLoRA Code Implementation
- QLoRA Process Analysis
- QLoRA Application Value
- QLoRA Questions and Thoughts
- QLoRA Details Supplement
<!-- more -->
## 1. Background Introduction: QLoRA / Base Model / Dataset
- QLoRA
    - QLoRA (Quantized Low-Rank Adaptation) fine-tuning method
    - Paper: https://arxiv.org/abs/2305.14314
    - QLoRA combines 4-bit quantization and LoRA technology, with the specific implementation steps as follows:
        - 4-bit Quantization: Using the bitsandbytes library to implement 4-bit NormalFloat (NF4) quantization, compressing the pre-trained model weights to 4 bits, significantly reducing memory usage.
        - LoRA: Implementing LoRA through the peft library, adding low-rank adapters (e.g., rank r=16), updating only a small number of parameters.
        - Combined Technology: Loading the quantized model, attaching LoRA adapters, and using 16-bit (bfloat16) for forward/backward propagation calculations.
- Base Model: DistilGPT-2
    - https://huggingface.co/distilbert/distilgpt2
- Alpaca Instruction Dataset
    - Official Version
        - https://huggingface.co/datasets/tatsu-lab/alpaca
        - Generated outputs using OpenAI's text-davinci-003 model; this dataset may contain errors or biases, and users are advised to use it cautiously and consider filtering methods, as mentioned in the dataset description.
    - yahma/alpaca-cleaned 
        - https://huggingface.co/datasets/yahma/alpaca-cleaned
        - Cleaned version that fixes hallucinations, incorrect answers, etc., providing high-quality data.
    - vicgalle/alpaca-gpt4
        - https://huggingface.co/datasets/vicgalle/alpaca-gpt4
        - Based on Alpaca prompts, generated outputs using GPT-4, but not mentioned as cleaned, may contain uncorrected errors.

## 2. QLoRA Code Implementation
- Load Model
- Preparing Dataset
- Fine-Tuning
- Save Trained Model

### 2.1. Load Model
```python
# Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Enable 4-bit quantization, replacing the model's linear layers (Linear / Conv1D) with quantized Linear4bit layers.
    bnb_4bit_use_double_quant=True, # Enable nested quantization to further compress quantized parameters, reducing storage overhead (Linear4bit internal computation logic).
    bnb_4bit_quant_type="nf4", # There are two formats for 4-bit quantization (nf4 and fp4), where nf4 is optimized based on normal distribution and usually performs better.
    bnb_4bit_compute_dtype=torch.bfloat16 # Set the data type for computation; actual weights are stored in 4-bit but mapped to bfloat16 for computation, meaning that the intermediate calculations in Linear4bit use bfloat16.
)
```

```python
# Select distilbert/distilgpt2 as the base model
model_id = "distilbert/distilgpt2"

# Load the entire model onto GPU 0
device_map = {"": 0}

# Load the original model
original_model = AutoModelForCausalLM.from_pretrained(model_id)

# Load the quantized model (applying the quantization configuration to the model)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    use_cache = False)
```

```python
# Load the tokenizer corresponding to the model and set the padding token to the end token
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
```

### 2.2. Preparing Dataset
```python
# Select yahma/alpaca-cleaned as the dataset
dataset_name = "yahma/alpaca-cleaned"

# Load the dataset
full_dataset = load_dataset(dataset_name, split="train")

# Select a small subset (1000 entries)
small_subset = full_dataset.shuffle(seed=42).select(range(1000))
```

```python
# Define the Alpaca dataset's prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Define the formatting_prompts_func function
def formatting_prompts_func(examples):

    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]

    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output)
        texts.append(text)
    return { "text" : texts }


# Apply the formatting_prompts_func function
small_subset = small_subset.map(formatting_prompts_func, batched=True,)
```

```python
# Apply tokenizer to "text"; if too long, truncate to the model's maximum length; pad all samples to the same length for batch training convenience
small_subset = small_subset.map(lambda samples: tokenizer(samples["text"], truncation=True, padding="max_length"), batched=True)
```

### 2.3. Fine-Tuning
```python
# LoRA Parameter Configuration
peft_config = LoraConfig(
    r=8, # Rank, larger values increase expressive power but also increase parameters.
    lora_alpha=16, # Scaling factor.
    lora_dropout=0.05, # Dropout probability.
    target_modules=["c_attn", "c_proj", "c_fc"],  # Modules where LoRA needs to be inserted.
    bias="none", # Whether to train the bias term: No.
    task_type="CAUSAL_LM", # Task type: Causal Language Modeling.
)

# Training Parameter Configuration
training_args = SFTConfig(
    output_dir="outputs", # Output path.
    logging_steps=1, # How often to log.
    num_train_epochs=3, # Number of training epochs.
    per_device_train_batch_size=2, # Training batch size per device.
    per_device_eval_batch_size=2, # Evaluation batch size per device.
    gradient_accumulation_steps=5, # Gradient accumulation.
    gradient_checkpointing=True, # Enable gradient checkpointing.
    learning_rate=2e-4, # Learning rate.
    optim="adamw_8bit", # Optimizer.
    weight_decay=0.01, # Weight decay.
    max_grad_norm=0.3, # Gradient clipping.
    warmup_ratio=0.03, # Warmup ratio.
    fp16=not torch.cuda.is_bf16_supported(), # Use half-precision training.
    bf16=torch.cuda.is_bf16_supported(),
    dataset_text_field="text",
)

# Instantiate SFTTrainer
trainer = SFTTrainer(
    model=quantized_model,
    train_dataset=small_subset,
    peft_config=peft_config,
    args=training_args,
)

# Output the number of trainable parameters
trainer.model.print_trainable_parameters()

# Start training
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

## 3. QLoRA Process Analysis
- Comparison of Original Model and Quantized Model
- Dataset Processing Flow
- Calculation of Trainable Parameters

### 3.1. Comparison of Original Model and Quantized Model
- The number of parameters remains unchanged (still the same number of matrix elements)
  - 81912576
- Parameter precision and size have changed (represented in 4-bit)
  - Parameter size changes
    - Original size: 318.47 MB
      - Estimate: 81912576 * 4 bytes / (1024^2) = 308.66 MB
    - Quantized size: 101.49 MB
      - Estimate
        - Number of parameters at 0.5 bytes: 42467328 = 6 layers * (768 * 2304 + 768 * 768 + 2 * 768 * 3072)
        - Number of parameters at 2 bytes: 39445248 = 81912576 - 42467328
        - (0.5 * 42467328 + 2 * 39445248) / (1024^2) = 95.49 MB
  - Model structure changes
    - attn
      - (c_attn): Conv1D(nf=2304, nx=768) -> (c_attn): Linear4bit(in_features=768, out_features=2304, bias=True)
      - (c_proj): Conv1D(nf=768, nx=768) -> (c_proj): Linear4bit(in_features=768, out_features=768, bias=True)
    - mlp
      - (c_fc): Conv1D(nf=3072, nx=768) -> (c_fc): Linear4bit(in_features=768, out_features=3072, bias=True)
      - (c_proj): Conv1D(nf=768, nx=3072) -> (c_proj): Linear4bit(in_features=3072, out_features=768, bias=True)
  - Parameter precision changes
    - Before quantization:
      - All parameters are dtype=torch.float32 (32-bit)
        - transformer.h.0.attn.c_attn.weight: torch.Size([768, 2304]), dtype=torch.float32
        - transformer.h.0.attn.c_proj.weight: torch.Size([768, 768]), dtype=torch.float32
        - transformer.h.0.mlp.c_fc.weight: torch.Size([768, 3072]), dtype=torch.float32
        - transformer.h.0.mlp.c_proj.weight: torch.Size([3072, 768]), dtype=torch.float32
    - After quantization:
      - dtype=torch.uint8 (4 places in each layer become 4-bit) actually stored using compression techniques, combining 2 4-bit weights into int8
        - transformer.h.0.attn.c_attn.weight: torch.Size([884736, 1]), dtype=torch.uint8
        - transformer.h.0.attn.c_proj.weight: torch.Size([294912, 1]), dtype=torch.uint8
        - transformer.h.0.mlp.c_fc.weight: torch.Size([1179648, 1]), dtype=torch.uint8
        - transformer.h.0.mlp.c_proj.weight: torch.Size([1179648, 1]), dtype=torch.uint8
      - dtype=torch.float16 (the remaining parameters become float16)

Changes Explanation
- Changes
  - Before quantization: transformer.h.0.attn.c_attn.weight: torch.Size([768, 2304]), dtype=torch.float32
  - After quantization: transformer.h.0.attn.c_attn.weight: torch.Size([884736, 1]), dtype=torch.uint8
- Explanation
  - The original float32 matrix [768, 2304] → The total number of parameters is 768 * 2304 = 1,769,472
  - Representing it in 4-bit means 1,769,472 * 0.5 byte = 884,736 bytes = 884736 uint8 (stored as packed uint8, where each byte stores two 4-bit weights)
  - [884736, 1] is the result of unfolding the original weights into one dimension and storing them in quantized form.

### 3.2. Dataset Processing Flow
- 1. Load the dataset
    - The dataset has 3 columns
        - instruction
        - input
        - output
    - The dataset contains 51760 entries
```
Dataset({
    features: ['output', 'input', 'instruction'],
    num_rows: 51760
})
```
- 2. Select a small subset
    - Randomly select 1000 entries
```
Dataset({
    features: ['output', 'input', 'instruction'],
    num_rows: 1000
})
```
- 3. Concatenate the 'output', 'input', and 'instruction' fields into a string, as the 'text' field
```
Dataset({
    features: ['output', 'input', 'instruction', 'text'],
    num_rows: 1000
})
```
- 4. Tokenize the "text" field using the tokenizer, generating "input_ids" and "attention_mask" fields
```
Dataset({
    features: ['output', 'input', 'instruction', 'text', 'input_ids', 'attention_mask'],
    num_rows: 1000
})
```

### 3.3. Calculation of Trainable Parameters
- Base model parameter count
  - 81912576
- Target modules in the model structure
  - attn
    - (c_attn): Conv1D(nf=2304, nx=768) -> (c_attn): Linear4bit(in_features=768, out_features=2304, bias=True)
    - (c_proj): Conv1D(nf=768, nx=768) -> (c_proj): Linear4bit(in_features=768, out_features=768, bias=True)
  - mlp
    - (c_fc): Conv1D(nf=3072, nx=768) -> (c_fc): Linear4bit(in_features=768, out_features=3072, bias=True)
    - (c_proj): Conv1D(nf=768, nx=3072) -> (c_proj): Linear4bit(in_features=3072, out_features=768, bias=True)
- Relationship between parameters
  - Total parameter count 82,502,400 - QLoRA trainable parameter count 589,824 = Base model parameter count 81912576
  - QLoRA trainable parameter count 589,824 = 6 layers * (24576 + 12288 + 30720 + 30720) = 6 * 98304
    - A total of 6 layers, each with 4 target modules
      - c_attn: 768 * 8 + 2304 * 8 = 24576
      - c_proj: 768 * 8 + 768 * 8 = 12288
      - c_fc: 3072 * 8 + 768 * 8 = 30720
      - c_proj: 768 * 8 + 3072 * 8 = 30720

## 4. QLoRA Application Value
### 4.1. Comparison of QLoRA and Full Parameter Fine-Tuning Methods

- Fine-tuning parameter count:  
  - QLoRA: Only fine-tunes a small portion of parameters (LoRA Adapter), in this case, the trainable parameters account for only 0.7149% of the total parameters.
  - Full parameter fine-tuning: Updates all model parameters, requiring more memory and computational resources.
- Memory Usage:  
  - QLoRA: Uses 4-bit quantization, significantly reducing memory usage.
  - Full parameter fine-tuning: Requires full precision (e.g., 16-bit or 32-bit), usually needing larger GPU memory.
- Training Speed:  
  - QLoRA: Due to fewer parameters and lower precision, training is faster.  
  - Full parameter fine-tuning: Updates all weights, resulting in longer training times.
- Performance:  
  - QLoRA: Due to quantization and limited parameter updates, performance may be slightly lower than full parameter fine-tuning, but still retains most capabilities.
  - Full parameter fine-tuning: Since all weights are optimized, performance may be higher.
- Applicable Scenarios:  
  - QLoRA: Suitable for resource-constrained environments or rapid experimentation.  
  - Full parameter fine-tuning: Suitable for resource-rich environments where maximum precision is required.

### 4.2. Advantages and Potential Limitations of QLoRA in This Task

Advantages:
- Memory Efficiency: 4-bit quantization and LoRA reduce memory usage, allowing fine-tuning to run on smaller GPUs (e.g., Colab free version).
- Speed: Due to fewer parameters and lower precision, training speed is faster, suitable for rapid iteration.
- Retaining Pre-trained Knowledge: Freezing most weights retains the generalization ability of the base model while adapting to the task.
- Suitable for Small Datasets: Performs well on small datasets like the 1,000-entry Alpaca subset, reducing the risk of overfitting.

Limitations:
- Performance Trade-off: Quantization and LoRA may lead to slightly lower performance than full parameter fine-tuning, especially in complex tasks.  
- Task Specificity: LoRA adapters are task-specific; switching tasks requires retraining or maintaining multiple adapters.  
- Quantization Noise: 4-bit precision introduces noise, potentially affecting output quality.  
- Hyperparameter Sensitivity: Adjusting `r`, `lora_alpha`, and quantization settings requires experimentation, increasing complexity.

## 5. QLoRA Questions and Thoughts
### 5.1. Is the only difference between LoRA and QLoRA fine-tuning the quantization of the model before training?

Yes, the main difference is whether the model being trained has been quantized beforehand.
- LoRA fine-tunes a pre-trained model that is not quantized, with weights typically in float32.
- QLoRA fine-tunes a pre-trained model that is quantized, such as 4-bit or 8-bit. The weights are in quantized uint8 or int8 format.

Memory and Inference Efficiency:
- LoRA occupies more memory during fine-tuning because it uses a full-precision pre-trained model (usually float32).
- QLoRA's advantage lies in significantly reduced memory usage since the model has already been quantized to 4-bit. Even large models can run in limited memory.

Similarities:
- Both LoRA and QLoRA fine-tune the original model by training a low-rank adaptation matrix without modifying the original pre-trained weights.
- Both only train the adaptation layers, resulting in relatively few parameters, making them suitable for resource-limited scenarios.

LoRA (Low-Rank Adaptation) Fine-Tuning Steps:
- Load a pre-trained model (usually with float32 weights).
- Freeze the original weights and only train the LoRA adaptation matrix.
- During fine-tuning, adapt the model to specific tasks by training the LoRA layers.
- During inference, use the original weights plus the trained LoRA layer's adaptation matrix.

QLoRA (Quantized LoRA) Fine-Tuning Steps:
- Quantize the pre-trained model: Load a model that has already been quantized, such as a 4-bit or 8-bit model.
- Freeze the quantized weights and only train the LoRA adaptation matrix.
- Fine-tune on the quantized model, optimizing the LoRA layer's adaptation matrix.
- During inference, still use the quantized weights plus the trained LoRA adaptation matrix.

### 5.2. Why is quantization usually only applied to linear layers?
The reasons can be explained from several perspectives: mathematics, engineering implementation, and model impact.

- 1. Linear layers have the most parameters and the highest computational load (parameter concentration).
  - Layer normalization (LayerNorm), activation functions (GELU), Dropout, etc., contain almost no weights or only a very small number of parameters.
  - Therefore, prioritizing the quantization of linear layers yields the greatest benefits and minimally impacts the model structure.
- 2. Linear layer structures are simple and easy to quantize and decode (simple mathematical structure).
  - Linear layers are among the few that can quantize weights + activations + gradients simultaneously (if needed), making them the main battleground.
  - Quantization typically involves two types:
    - Weight Quantization: Storing weights in low precision (most common).
    - Activation Quantization: Quantizing intermediate values (more complex).
  - Linear transformations have a very clear, fixed mathematical structure, suitable for being:
    - Encoded as 4-bit / 8-bit arrays.
    - Decoded using scale + zero_point for recovery.
    - Accelerated using kernel-fusion (e.g., CUDA implementations provided by bitsandbytes).
- 3. Non-linear operations are difficult to quantize (other layers are not suitable; quantizing linear layers has minimal impact).
  - These layers either contain no parameters or exhibit significant behavioral changes; quantizing these parts yields minimal benefits and increases complexity and error.
    - Non-linear functions (GELU, ReLU) have no fixed weights, making them difficult to encode in advance.
    - Normalization (LayerNorm, BatchNorm) involves dynamically calculating means, variances, and divisions, making static quantization challenging.
    - Control structures (Dropout, Mask) behave differently during training/inference, making quantization less meaningful.
- 4. Engineering has already highly optimized linear layer quantization (engineering optimization).
  - Existing frameworks have focused on optimizing operators like Linear and Conv, with poor or no support for non-linear operations.
    - bitsandbytes
    - Intel Neural Compressor
    - NVIDIA TensorRT
    - ONNX Runtime + QNNPACK

### 5.3. Why is the lm_head layer not quantized, even though it is also a linear layer?

In theory, this layer could also be quantized, but it is not quantized by default.

The reasons are mainly as follows:

- 1. lm_head is used for output → Precision impact is more sensitive (precision sensitivity).
  - lm_head is the last layer of the model, used to project hidden states into the vocabulary space, outputting logits.
  - The output of this layer directly affects:
    - The distribution of softmax.
    - The token ordering of predictions.
    - The quality of the generated text.
  - Therefore, it is very sensitive to precision. A slight weight error may lead to incorrect token ordering, generating completely different results.
- 2. Most quantization frameworks skip lm_head by default (framework default behavior, limited space savings).
  - For example, bitsandbytes, AutoGPTQ, etc., by default do not perform 4-bit quantization on lm_head when quantizing Transformer models because:
    - This layer usually has only one instance (unlike attention and MLP, which have many layers).
    - The precision impact is greater.
    - Space savings are limited (relative to the entire model), making quantization less meaningful.
- 3. It may be used for weight sharing (tie_weights) (weight sharing, bound to embedding weights, quantization may be incompatible).
  - In some models, lm_head.weight shares weights with the embedding layer wte.weight (tie-weights).
  - In this case, quantizing lm_head would affect the embedding → encoding → decoding entire loop.
  - If you quantize lm_head but do not synchronize the quantization of wte, or vice versa, it may lead to inconsistencies or even errors.

### 5.4. Why do non-linear layers change dtype to float16 after quantization?

Although these layers do not explicitly set their dtype, they will automatically be converted to a lighter precision when load_in_4bit=True.

This is actually part of the transformers' AutoPrecision inference mechanism (automatic scheduling), used in conjunction with bitsandbytes.

### 5.5. What changes occurred in the precision of the quantized model?

- The original model's precision is float32 (disk).
- The weights of the quantized model are stored as int4 (packed uint8) (memory).
- Input float32/bfloat16 (externally passed).
- Weights int4 are temporarily decoded into bfloat16 for computation.
- Double quant → further compression of scale, reducing burden.
- Output bfloat16 → Subsequent layers can continue executing in low precision.

Note 1: Only the storage of linear layers is 4-bit; the storage, computation, and decoding output of other layers are all bfloat16/float16.

Note 2: Inference does not change the model's parameters; only during backpropagation do they change. During inference, the 4-bit weights are temporarily converted to bfloat16 for computation, and the output is also of bfloat16 type.

Note 3: Fine-tuning a quantized model (e.g., LoRA + 4-bit) also only updates a portion of float parameters (like the LoRA adapter); the 4-bit weights themselves are not directly modified.

### 5.6. What is bnb_4bit_compute_dtype=torch.bfloat16 for, and how do computation precision and storage precision differ?
By default, even if you quantize parameters, the computation is still in float32.

Therefore, you can set the computation precision to torch.bfloat16 or torch.float16 to speed up the computation process and reduce memory usage.

This computation precision dtype is set during the internal runtime of Linear4bit and is part of the internal computation logic.

### 5.7. Temporary Quantization vs. Permanent Storage?
- Temporary Quantization
  - Loads the original float32 weight model and quantizes it in memory based on the quantization configuration.
  - During inference, the 4-bit weights are temporarily converted to bfloat16 for computation.
  - Key points:
    - The 4-bit weights of the quantized model only exist temporarily in memory; they are not permanently saved.
    - The original model file on disk is not changed; it only affects the computation method in memory.
- Permanent Storage
  - Saves the quantized version of the model, loading the quantized model directly without needing to apply the quantization configuration.

### 5.8. What precision does QLoRA use to store the trained model?
QLoRA has two common strategies for quantization and storage during training:
- Situation 1: Quantized during training, but retains the original model's float32 weights for storage.
  - The original model's weights will still be saved in float32 format, rather than in the quantized format.
  - The quantization operation is usually applied temporarily during inference; the quantized weights during training are just to save memory and speed up computation but do not necessarily need to be stored permanently.
  - Storing the original model in float32 format ensures better model compatibility and ease of use in the future.
  - LoRA adaptation layers: The weights of the LoRA layers are still in float32 format and will be stored alongside the original model.
  - The model saved by QLoRA is:
    - Stores the original model's weights in float32 format.
    - Stores the weights of the LoRA adaptation layers, which remain in float32 precision.
- Situation 2: Both quantized during training and storage.
  - This method is mainly used to save storage space and speed up inference.
  - The model saved by QLoRA is:
    - Stores the quantized model, i.e., 4-bit (usually uint8) weights.
    - Stores the weights of the LoRA adaptation layers, which remain in float32 precision.

Is it necessary to keep the data types of LoRA adaptation layer weights and model weights consistent? Answer: The data types do not need to be completely consistent for the following reasons:
- Independence of LoRA adaptation layers from original model weights.
- The inconsistency in data types between LoRA layer weights and quantized model weights does not cause conflicts.
  - During inference, the quantized weights (e.g., 4-bit) will be restored to float16 or bfloat16 format for computation, while the weights of the LoRA layers (float32) still participate in the computation; the inconsistency in precision will not cause issues.
  
## 6. QLoRA Details Supplement
### 6.1. device_map
device_map is a dictionary used to inform the model loader (e.g., AutoModel.from_pretrained in transformers) which devices to load different parts of the model onto. The keys typically represent a part of the model (e.g., the name of a layer or module), and the values represent the device number or device name (e.g., GPU index 0, 1, or "cpu").

```
device_map={"": 0}  # Load the entire model onto GPU 0 (device number 0) (an empty string indicates the entire model; in PyTorch, 0 usually corresponds to the first GPU (i.e., cuda:0)).
```
```
device_map = {
    "transformer.layer.0": 0,  # Load layer 0 onto GPU 0
    "transformer.layer.1": 1   # Load layer 1 onto GPU 1
}
```
```
device_map = "auto" # Automatically allocate the model based on available devices (requires support from the accelerate library).
```

Extension: How to check available GPU devices
```
import torch
print(torch.cuda.is_available())  # Check if there is a GPU
print(torch.cuda.device_count())  # Check the number of GPUs
print(torch.cuda.current_device())  # Current default GPU index
``` 

### 6.2. seed=42 is a common "magic number"
42 is a conventional value, originating from "The Hitchhiker's Guide to the Galaxy," representing "the ultimate answer to life, the universe, and everything." Of course, you can use 123 or any other value; as long as you use the same seed each time, you can reproduce the results.

## 7. Reference
- huggingface SFTTrainer: https://huggingface.co/docs/trl/v0.7.4/en/sft_trainer
- google: https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
- unsloth: https://huggingface.co/blog/Andyrasika/finetune-unsloth-qlora 