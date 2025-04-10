---
icon: lightbulb
sidebar: false
date: 2025-04-09
prev: ./033_sft_trainer_sourcecode_prepare_dataset
next: ./031_qlora
category:
  - LLM
tag:
  - SFTTrainer
  - Sourcecode
  - Prepare Model
---
# SFTTrainer Sourcecode -- Prepare Model
<!-- more -->
## 1. Prepare Model Overall Logic  
Overall Logic  
- 1. Load the base model based on `model_id` or `model_path`.  
- 2. If `peft_config` exists, prepare the PEFT model based on `LoraConfig`.  
- 3. Determine whether it is QLoRA (check if the model attributes `is_loaded_in_4bit` or `is_loaded_in_8bit` are `True`).  
- 4. If it is QLoRA, freeze the parameters of the base model and convert all non-INT8 type parameters to `fp32`.  
- 5. Instantiate the PEFT model `PeftModelForCausalLM`.  
- 6. Instantiate the `LoraModel`.  
    - Verify whether the `lora_config.target_modules` configuration has corresponding modules. If `lora_config.target_modules` is not configured, check if there are default `target_modules` for the model type in `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`. If not, raise an error: "Please specify `target_modules` in `peft_config`".  
    - Update the target modules `target_modules` based on `lora_config` to obtain the updated modules with `bnb.nn.Linear4bit`.  
        - Retrieve the `in_features` and `out_features` of the base layer `bnb.nn.Linear4bit`.  
        - Generate adapter layers: `lora_A` and `lora_B`.  
            - `self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)`  
            - `self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)`  
        - Activate the adapter layer: `layer.requires_grad_(True)`.  
- 7. If `args.bf16 == True` and `model.is_loaded_in_4bit == True`, convert the weights of some modules to `bfloat16`.  

## 2. Prepare Model Code Details  
### 2.1. SFTTrainer.__init__
```python
class SFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.
    """

    def __init__():
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        # PEFT configuration and model wrapping
        if peft_config is not None:
            model = self._prepare_peft_model(model, peft_config, args)
```

### 2.2. _create_model_from_path
```python
def _create_model_from_path():
    """Creates a model from a path or model identifier."""

    # Create model
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
    return model
```

### 2.3. _prepare_peft_model 
```python
def _prepare_peft_model():
    """Prepares a model for PEFT training."""

    is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

    # Prepare model for kbit training if needed
    if is_qlora and not is_sharded_qlora:
        model = self._prepare_model_for_kbit_training(model, args)
        # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
        args = dataclasses.replace(args, gradient_checkpointing=False)
    elif args.gradient_checkpointing:
        model = self._enable_gradient_checkpointing(model, args)

    # Create PEFT model
    model = get_peft_model(model, peft_config)

    # Handle bf16 casting for 4-bit models
    if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
        peft_module_casting_to_bf16(model)

    return model
```

### 2.4. prepare_model_for_kbit_training
```python
def prepare_model_for_kbit_training():
    # freeze base model's layers
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (
            (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
        ) and param.__class__.__name__ != "Params4bit":
            param.data = param.data.to(torch.float32)
    
    return model
```

### 2.5. get_peft_model
- peft_config.task_type="CAUSAL_LM"
- peft_config.is_prompt_learning=False

```python
MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, type[PeftModel]] = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
    "QUESTION_ANS": PeftModelForQuestionAnswering,
    "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
}
```
```python
def get_peft_model():
    """
    Returns a Peft model object from a model and a config, where the model will be modified in-place.
    """

    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        model,
        peft_config,
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
```

### 2.6. PeftModelForCausalLM.__init__
```python
class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.
    """

    def __init__(
        self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
```

### 2.7. PeftModel.__init__
- peft_config.peft_type=PeftType.LORA
```python
class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        super().__init__()  

        cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
        ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
        with ctx():
            self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)

        self.set_additional_trainable_modules(peft_config, adapter_name)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model) 
```

### 2.8. LoraModel.__init__
```python
class LoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.
    """
    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        if lora_config._custom_modules:
            # Experimental custom LoRA module support. Allows users to pass a custom mapping for unsupported layer
            # types by impelementing their own LoRA layers.
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None

                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target

                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break

                return new_module

            dispatchers.append(dynamic_dispatch_func)

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_eetq,
                dispatch_aqlm,
                dispatch_awq,
                dispatch_gptq,
                dispatch_hqq,
                dispatch_torchao,
                dispatch_megatron,
                dispatch_default,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`."
            )

        return new_module
```

```python
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj", "fc1", "fc2"],
    "gemma": ["q_proj", "v_proj"],
    "gemma2": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
}
```

Extension: How to view the model type

```python
from transformers import AutoModel, AutoConfig
model_id = "distilbert/distilgpt2"
config = AutoConfig.from_pretrained(model_id)
print(config.model_type) # result：gpt2
```

### 2.9. dispatch_bnb_4bit
```python
def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
    if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        fourbit_kwargs = kwargs.copy()
        fourbit_kwargs.update(
            {
                "compute_dtype": target_base_layer.compute_dtype,
                "compress_statistics": target_base_layer.weight.compress_statistics,
                "quant_type": target_base_layer.weight.quant_type,
            }
        )
        new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

    return new_module
```

### 2.10. Linear4bit.__init__
```python
class Linear4bit(torch.nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.fan_in_fan_out = False

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
```

### 2.11. LoraLayer.__init__(self, base_layer)
```python 
class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)
```

### 2.12. peft_module_casting_to_bf16
```python
def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
```