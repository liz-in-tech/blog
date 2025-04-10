---
icon: lightbulb
sidebar: false
date: 2025-02-14
prev: ./020_neo4j
next: ./018_huggingface
category:
  - LLM
tag:
  - Ollama
---
# Ollama User Guide
- About
- Features
- How To Use
<!-- more -->
## 1. About
Official Website: https://ollama.com/

Github: https://github.com/ollama/ollama

Model Library: https://ollama.com/search

Model Library Information:
- Model Categories: All/Embedding/Vision/Tools
- Model Sorting: Popular/Newest
- Model Name
- Model Description
- Available Parameter Scales
- Model Size, Unique Identifier, Upload Time, and Download Command for Different Scales and Variants
- Download Count (Indicating Popularity)
- Number of Tags (Different Scales and Variant Versions Available for Download)
- Update Date

## 2. What
Used for local deployment of open-source large models.

Developers can test and debug models locally.

Users can use language models offline to ensure data privacy, fast response times, and complete freedom, all for free.

Features:
- Simple operation, simplified model deployment and management, with a low usage threshold.
- Supports multiple platforms: macOS, Linux, and Windows.
- The model library is comprehensive and updated quickly. In addition to models in the library, users can download GGUF models from HuggingFace, deploy locally downloaded GGUF model files, or create custom models.

## 3. How To Use (macOS Platform)
### 3.1. Installing Ollama
1. Download from the official website to get Ollama-darwin.zip. Double-click to get Ollama.app, and move it to Applications.
2. The first time you open it, you will be prompted to install the command-line tools. Enter the password to authorize, and installation will be complete.

### 3.2. Downloading Models
#### 3.2.1. From the Model Library
1. Find the model you want to download locally from the model library (it will be downloaded to the hidden .ollama folder in the user directory).
2. Copy the download command to the command line to start the download (usually a `run` or `pull` command).

#### 3.2.2. From HuggingFace
1. Search for GGUF models on HuggingFace (https://huggingface.co/models).
2. Execute the command in the terminal:
```
ollama run hf.co/{username}/{repository}
ollama run hf.co/{username}/{repository}:{quantization} 
```
### 3.3. Creating a Custom Model and Pushing to the Model Library
1. Create the model file `Modelfile`
- FROM: Base Model
    - FROM ./vicuna-33b.Q4_0.gguf # Ollama supports importing GGUF models in the Modelfile by pointing to the local file path of the GGUF model to import.
    - FROM llama3.2
- PARAMETER: Parameter Configuration
- SYSTEM: System Prompt
```
FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```

2. Execute the command to create the custom model
```
ollama create my_model_name -f ./Modelfile
```

3. Run the custom model
```
ollama run my_model_name
```

4. Upload to the official model library:

   1) Register and log in to the official website, where the name will be the namespace for uploading the model (`my_namespace`).
   2) Go to the "My models" section and create a new model. The model name should match the custom `my_model_name`. Follow the prompt to set up the public key in the website Settings.
   3) Copy the model to the namespace:
   ```
   ollama cp my_model_name my_namespace/my_model_name
   ```
   4) Push the model to the model library:
   ```
   ollama push my_namespace/my_model_name
   ```
   5) Once the upload is successful, if you choose to make the model public, it will appear in the model library.

### 3.4. Managing and Using Models
- Add:
    - pull
    - run
    - cp
    - create
- Remove:
    - rm
- Check:
    - List models: list
    - List running models: ps
    - Show details: show
- Usage:
    - run
    - Exit chat: Use Ctrl + d or /bye to exit
    - stop

### 3.5. More Commands
Type `ollama` in the command window to see all usage options.
```
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama （和运行桌面应用是一样的）
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
```