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
# Ollama操作指南
- 是什么
- 作用与特点
- 怎么使用
<!-- more -->
## 1. About
官网：https://ollama.com/

Github：https://github.com/ollama/ollama

模型库：https://ollama.com/search

模型库显示信息
- 模型分类：All/Embedding/Vision/Tools
- 模型排序：Popular/Newest
- 模型名称
- 模型描述
- 可选的参数量规模
- 不同规模和变体版本的模型大小、唯一标识、上传时间和下载命令
- 被下载次数（表示了受欢迎程度）
- Tags数（可下载的不同规模和变体版本的模型个数）
- 更新日期

## 2. What
用于本地部署开源大模型

开发者可以在本地测试和调试模型

用户可以在不联网的情况下使用语言模型，确保数据隐私，响应速度快，完全免费

特点
- 操作简单，简化了模型部署和管理，使用门槛低
- 支持多平台：macOS, Linux, and Windows
- 模型库很全面且更新很快，除了模型库已有的模型，用户可以在HuggingFace下载gguf模型，可以部署本地已下载好的gguf模型文件，也可以创建自定义模型

## 3. How To Use（macOS平台操作）
### 3.1. 安装Ollama
1.在官网下载，得到Ollama-darwin.zip，双击得到Ollama.app，移入Applications  
2.第一次打开会提示安装命令行工具，输入密码授权，安装完成
### 3.2. 下载模型
#### 3.2.1. 从模型库下载
1.从模型库找到想要下载到本地的模型（下载到了用户目录下的.ollama隐藏文件夹下）  
2.复制下载命令到命令行后开始下载（一般为run或pull命令）
#### 3.2.2. 从HuggingFace下载
1.从HuggingFace（https://huggingface.co/models）中搜索GGUF模型  
2.在命令行执行命令
```
ollama run hf.co/{username}/{repository}
ollama run hf.co/{username}/{repository}:{quantization} 
```
### 3.3. 创建自定义模型并push到模型库
1.创建模型文件Modelfile
- FROM：基础模型
    - FROM ./vicuna-33b.Q4_0.gguf # Ollama 支持在 Modelfile 中导入 GGUF 模型，指向要导入的GGUF模型的本地文件路径
    - FROM llama3.2
- PARAMETER：参数配置
- SYSTEM：系统指令 system prompt
```
FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```
2.执行创建自定义模型命令
```
ollama create my_model_name -f ./Modelfile
```
3.运行自定义模型
```
ollama run my_model_name
```
4.上传到官网模型库 

1）注册并登录官网，此处的名称即为上传模型的命名空间my_namespace  
2）进入My models界面New一个模型, 模型名和刚才自定义的my_model_name一致，按提示在官网Settings设置公钥  
3）把模型复制到命名空间下
```
ollama cp my_model_name my_namespace/my_model_name
```
4）push模型到模型库
```
ollama push my_namespace/my_model_name
```
5）上传成功，如果选择公开模型的话，在模型库中可以搜到自己的模型了
### 3.4. 管理并使用模型
- 增
    - pull
    - run
    - cp
    - create
- 删
    - rm
- 查
    - 查列表：list
    - 查运行中的模型列表：ps
    - 查详情：show
- 使用
    - run
    - 退出聊天：Use Ctrl + d or /bye to exit
    - stop
### 3.5. 更多命令
命令窗口输入 ollama 即可看到所有用法
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