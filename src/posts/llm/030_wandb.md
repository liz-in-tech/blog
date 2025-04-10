---
icon: lightbulb
sidebar: false
date: 2025-03-10
prev: ./031_qlora
next: ./029_unsloth_grpo
category:
  - LLM
tag:
  - wandb
---
# Lightweight Visualization Tool for Deep Learning: wandb
<!-- more -->
## 1. What is wandb?
wandb (Weights & Biases) is a model training visualization platform similar to TensorBoard. However, compared to TensorBoard, wandb is more powerful, mainly in the following aspects:
- Four core functions (wandb is not just a data visualization tool. It has more powerful model and data version management. Additionally, it can optimize the models you train.)
    - Dashboard: Track the training process and provide visual results.
    - Reports: Save and share some details and valuable information during training.
    - Sweeps: Use hyperparameter tuning to optimize your trained model.
    - Artifacts: Dataset and model version control, allowing you to build your own pipeline to save datasets, models, and evaluation results.
- Strong compatibility (compatible with various frameworks and environments): It can be used with Jupyter, TensorFlow, PyTorch, Keras, Scikit, fast.ai, LightGBM, XGBoost, and can work with various machine learning infrastructures: AWS, GCP, Kubernetes, Azure, and local machines.
- Model reproducibility: wandb is more conducive to model reproducibility. This is because wandb not only records metrics but also records hyperparameters and code versions.
- Automatic cloud upload: wandb allows you to conveniently view all the models you have created, so you don't have to spend a lot of time rerunning old experiments.
- Fast and flexible integration: You can add wandb to your project in just 5 minutes. Download the free open-source Python package of wandb, then insert a few lines of code, and every time you run the model, you will get well-documented metrics and records.
- Centralized dashboard: wandb provides the same centralized dashboard. No matter where you train the model, whether on a local machine, lab cluster, or cloud instance, you don't have to spend time copying TensorBoard files from other machines.
- Powerful tables: Search, filter, classify, and group results of different models. You can easily view thousands of model versions and find the best model for different tasks.
- Analyze system hardware conditions such as CPU and GPU usage while running.

![](../../assets/030_wandb.png)

## 2. Common Functions
- `wandb.init()`: Initialize
    - `project`: Project name
- `wandb.config`: Hyperparameter configuration
- `wandb.log()`: Record continuously changing metrics in the training loop
- `wandb.Image()`: Image display
- `wandb.save()`: Save model
- `wandb.watch()`

## 3. Usage
### 3.1. Install wandb Library
```python
!pip install wandb
```

### 3.2. Register on the Official Website to Get API Key
Official website: [https://wandb.ai/](https://wandb.ai/)

You can generally register with a Google account or GitHub account. Remember to save your username during registration. After successful registration, registration information will be displayed on this page, and the API key can also be obtained from this page.

### 3.3. Run `wandb login` in Terminal
```python
wandb.login()
```
Enter the API key.

Automatic login setup: After running the `wandb login` command, a `.netrc` file will be generated under `/root/.netrc`. Modify this file, where the password is the API key obtained in the second step, and you won't need to enter the password again.

### 3.4. Initialization
```python
import wandb

wandb.init(project="my-project")
```

### 3.5. Declare Hyperparameters
```python
wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128
```

### 3.6. Log Records
```python
def my_train_loop():
    for epoch in range(10):
        loss = 0 # change as appropriate :)
        wandb.log({'epoch': epoch, 'loss': loss})
```

### 3.7. Send Alerts
```python
acc_threshold = 0.3
if accuracy <= acc_threshold:
        wandb.alert(
            title='Low Accuracy',
            text=f'Accuracy {accuracy} at step {training_step} is below the acceptable threshold, {acc_threshold}',
        )
```

### 3.8. Save Files
```python
wandb.save("mymodel.h5")
```

### 3.9. View wandb Dashboard
Open the link in the line with the rocket in the run results, and if the interface appears in the browser, it means it was successful. 