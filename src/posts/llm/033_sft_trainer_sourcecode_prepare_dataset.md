---
icon: lightbulb
sidebar: false
date: 2025-04-09
prev: ./034_sft_trainer_sourcecode_prepare_trainer
next: ./032_sft_trainer_sourcecode_prepare_model
category:
  - LLM
tag:
  - SFTTrainer
  - Sourcecode
  - Prepare Dataset
---
# SFTTrainer Sourcecode -- Prepare Dataset
<!-- more -->
## 1. Prepare Dataset Overall Logic
Overall Logic
- 1. If `processing_class` is None, use the base model's tokenizer
- 2. Process the Data collator by right-padding with `pad_token` to ensure consistent length
- 3. Check if the dataset column names contain "input_ids". If present, it indicates preprocessing has been done, and subsequent preprocessing steps will be skipped
- 4. If column names contain "input_ids" (indicating preprocessing is done), `formatting_func` will be ignored. Otherwise, process with `formatting_func`
    - Automatically determine whether to enable batch processing based on the return type of `formatting_func`, then map the dataset to format each sample as {"text": formatting_func result}
- 5. If the dataset column names contain "prompt" and "completion" fields
    - Determine whether the format is conversational (containing "role" and "content") or text-based
    - Map the dataset:
        - For conversational format, format each sample as {"messages": example["prompt"] + example["completion"]}
        - For text format, format each sample as {"text": example["prompt"] + example["completion"]}
- 6. Perform preprocessing (skip if "input_ids" is present in column names)
    - Convert conversational format to unified ChatML format: `{'messages': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is blue.'}]}`
    - Apply `tokenizer.apply_chat_template` to convert the "messages" conversational format to text format: `{"text": "xxx"}`
    - Tokenize the "text" field using the tokenizer to generate "input_ids" and "attention_mask" fields
- 7. Return the processed dataset, which must contain three fields: 'text', 'input_ids', 'attention_mask'

## 2. Prepare Dataset Code Details  
### 2.1. SFTTrainer.__init__
```python
class SFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.
    """
    def __init__():
        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)
        
        # Data collator
        if data_collator is None:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
            pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            data_collator = DataCollatorForLanguageModeling(pad_token_id)

        # Dataset
        train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
        if eval_dataset is not None:
            packing = args.packing if args.eval_packing is None else args.eval_packing
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(
                    eval_dataset, processing_class, args, packing, formatting_func, "eval"
                )
```

### 2.2. DataCollatorForLanguageModeling
```python
@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch if
    they are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    >>> from trl import DataCollatorForLanguageModeling
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [
    ...     {"input_ids": [1, 2, 3]},
    ...     {"input_ids": [4, 5]}
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[   1,   2,   3],
                          [   4,   5,   0]]),
     'attention_mask': tensor([[  1,   1,   1],
                               [  1,   1,   0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]
        labels = [torch.tensor(example["input_ids"]) for example in examples]

        # Pad
        output = {}
        output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="right")
        output["attention_mask"] = pad(attention_mask, padding_value=0, padding_side="right")
        output["labels"] = pad(labels, padding_value=-100, padding_side="right")

        return output
```

### 2.3. _prepare_dataset
```python
def _prepare_dataset():
    # If the dataset is already preprocessed (tokenized), skip the processing steps.
    column_names = list(next(iter(dataset)).keys())
    is_processed = "input_ids" in column_names

    # Apply the formatting function if any
    if formatting_func is not None and is_processed:
        warnings.warn(
            "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
            "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
            "`formatting_func` or pass a dataset that is not already processed.",
            UserWarning,
        )

    if formatting_func is not None and not is_processed:
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

        batched = isinstance(formatting_func(next(iter(dataset))), list)

        def _func(example):
            return {"text": formatting_func(example)}

        dataset = dataset.map(_func, batched=batched, **map_kwargs)

    # If the dataset is prompt-completion, convert it to language modeling type
    first_example = next(iter(dataset))
    if "prompt" in first_example.keys() and "completion" in first_example.keys():
        key = "messages" if is_conversational(first_example) else "text"

        def concat_prompt_completion(example):
            return {key: example["prompt"] + example["completion"]}

        dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])

    if not is_processed:
        # Convert the dataset to ChatML if needed
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
        column_names = next(iter(dataset)).keys()
        dataset = dataset.map(
            maybe_convert_to_chatml,
            remove_columns="conversations" if "conversations" in column_names else None,
            **map_kwargs,
        )

        # Apply the chat template if needed
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
        column_names = next(iter(dataset)).keys()
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={"tokenizer": processing_class},
            remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
            **map_kwargs,
        )

        # Tokenize the dataset if needed
        if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

        def tokenize(example, processing_class, dataset_text_field):
            processed = processing_class(text=example[dataset_text_field])
            if (
                processing_class.eos_token_id is not None
                and processed["input_ids"][-1] != processing_class.eos_token_id
            ):
                processed["input_ids"] = processed["input_ids"] + [processing_class.eos_token_id]
                processed["attention_mask"] = processed["attention_mask"] + [1]
            return processed

        dataset = dataset.map(
            tokenize,
            fn_kwargs={"processing_class": processing_class, "dataset_text_field": args.dataset_text_field},
            **map_kwargs,
        )
    
    return dataset
```

