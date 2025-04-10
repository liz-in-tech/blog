---
icon: lightbulb
sidebar: false
date: 2025-04-09
prev: false
next: ./033_sft_trainer_sourcecode_prepare_dataset
category:
  - LLM
tag:
  - SFTTrainer
  - Source Code
  - Prepare Train
---
# SFTTrainer Source Code Exploration: Prepare Train
- Prepare Train Overall Logic
- Prepare Train Code Details  
    - _inner_training_loop
    - training_step
    - compute_loss
    - PeftModelForCausalLM.forward
    - Linear4bit.forward
<!-- more -->
## 1. Prepare Train Overall Logic
Overall Logic
- Initialize SFTTrainer
- Execute trainer.train()
    - Execute inner_training_loop()
        - get_train_dataloader
        - Setting up training control variables
            - Number of training epochs: num_train_epochs
            - Number of update steps per epoch: num_update_steps_per_epoch
            - Total number of update steps: max_steps
                - If max_steps is not set, then max_steps == num_train_epochs * num_update_steps_per_epoch
            - Global batch size per update step: total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        - Begin running training
        - If args.eval_on_start is True, perform one validation before training
        - Epochs loop
            - Load data: train_dataloader
            - Update Steps loop (global_batch_size, i.e., the batch size for each parameter update)
                - Get batch data: batch_samples
                    - batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                - Batch loop (micro_batch_size)
                    - Forward pass: outputs = model(**inputs) # **inputs is the batch data
                        - PeftModelForCausalLM.forward
                            - inputs_embeds = self.word_embeddings(input_ids)
                            - return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
                        - Linear4bit.forward
                            - result = self.base_layer(x, *args, **kwargs)
                            - output = lora_B(lora_A(dropout(x))) * scaling
                            - result = result + output
                            - return result
                    - Calculate Loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    - Backward pass: self.accelerator.backward(loss, **kwargs)
                - Gradient accumulation and parameter update: self.optimizer.step()

Core Training Logic
```python
for epoch in range(epochs_trained, num_train_epochs): # Epoch loop
    for i, inputs in enumerate(batch_samples): # Batch loop
        tr_loss_step = self.training_step(model, inputs, num_items_in_batch) # Execute a single training step (forward pass, loss calculation, and backward pass), returns the loss value tr_loss_step for current batch
        tr_loss = tr_loss + tr_loss_step # Accumulate loss
        if do_sync_step: # Check if synchronization step is needed (do_sync_step indicates whether this is the final gradient accumulation step or the last step of the epoch)
            """
            Supports gradient accumulation, allowing gradients from multiple batches to be accumulated before parameter updates
            Optimizer updates are only triggered when do_sync_step is True, ensuring a balance between computational efficiency and memory usage
            """
            self.optimizer.step() # Update model parameters using accumulated gradients
            if not self.accelerator.optimizer_step_was_skipped: # Check if optimizer update was skipped (e.g., due to gradient overflow or NaN)
                self.lr_scheduler.step() # Update current learning rate (dynamic learning rate adjustment, only executed when optimizer updates successfully, maintaining consistency with training progress)
            model.zero_grad() # Clear gradients for next computation
            self.state.global_step += 1 # Increment global training step count
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate) # Log metrics, save model checkpoints, perform evaluation
```

## 2. Prepare Train Code Details  
### 2.1. SFTTrainer.__init__
```python
class SFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.
    """

    def __init__():
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **super_init_kwargs,
        )
```

### 2.2. Trainer.train()
```python
class Trainer
    def train():
        """
        Main training entry point.
        """

        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )
```

### 2.3. _inner_training_loop
```python
def _inner_training_loop():
    # Data loader and number of training steps
    train_dataloader = self.get_train_dataloader()

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0, device=args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()
    grad_norm: Optional[float] = None
    learning_rate = None

    if args.eval_on_start:
        self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    epochs_trained = 0

    for epoch in range(epochs_trained, num_train_epochs):
        epoch_dataloader = train_dataloader
        
        steps_in_epoch = (
            len(epoch_dataloader)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        
        step = -1
        epoch_iterator = iter(epoch_dataloader)
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = num_examples % args.gradient_accumulation_steps
        if remainder == 0:
            remainder = args.gradient_accumulation_steps
        update_step = -1
        total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
        if args.gradient_accumulation_steps == 1:
            total_updates -= 1
            
        for _ in range(total_updates):
            update_step += 1
            num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
            for i, inputs in enumerate(batch_samples):
                step += 1

                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch

                tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss = tr_loss + tr_loss_step

                if do_sync_step:
                    self.optimizer.step()

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
    
    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        self._load_best_model()
    
    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    train_loss = self._total_loss_scalar / effective_global_step

    return TrainOutput(self.state.global_step, train_loss, metrics)
```

### 2.4. training_step
```python
def training_step(self, model, inputs):
    """
        Perform a training step on a batch of inputs.
    """

    model.train()
    
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()
    
    inputs = self._prepare_inputs(inputs)
    
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
    
    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    if self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward() 
    else:
        # Finally we need to normalize the loss for reporting
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss, **kwargs) 

        return loss.detach()
```

### 2.5. compute_loss
```python
def compute_loss(self, model, inputs):
    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
    
    outputs = model(**inputs)

    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    
    return loss
```

### 2.6. PeftModelForCausalLM.forward
```python
class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.
    """

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_ids=None,
            **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # concat prompt labels
        if labels is not None:
            prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
```

### 2.7. Linear4bit.forward
```python
class Linear4bit(torch.nn.Module, LoraLayer):
    # Lora implemented in a dense layer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    output = lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, torch.nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    output = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )
                if requires_conversion:
                    output = output.to(expected_dtype)
                result = result + output

        return result
```