---
icon: lightbulb
sidebar: false
date: 2024-06-20
prev: ./007_computer_use
next: ./005_llama
category:
  - LLM
tag:
  - LLM
  - LeaderBoard
---
# LLM排行榜平台
  - LLM性能评估平台
  - LLM组织和模型 
<!-- more -->

## 1. LLM性能评估平台
### 1.1. LMSYS（匿名 AB Test）
组织:   
LMSYS 和 UC Berkeley SkyLab

评估方式:  
Chatbot Arena - 一个众包的随机对战平台。   
在真实世界中通过人类偏好评估 LLMs，向两个匿名模型（例如，ChatGPT、Gemini、Claude、Llama）提出任何问题，并为更好的一个投票！ 

评估结果：    
Arena Elo  
Elo 评级系统是以其发明者、匈牙利裔美国物理学家 Arpad Elo 的名字命名的。它最初是在 20 世纪 60 年代为评级国际象棋棋手开发的。

网址:  
[https://chat.lmsys.org/](https://chat.lmsys.org/)  
[https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)  

### 1.2. LiveBench (一直更新测试数据库来减少污染)
组织:  
Abacus.AI

特点:
- LiveBench 旨在通过每月发布新问题以及基于最近发布的数据集、arXiv 论文、新闻文章和 IMDb 电影梗概的问题来限制潜在的污染。
- 每个问题都有可验证的、客观的真实答案，允许对难题进行准确和自动的评分，而无需使用 LLM 作为评判者。
- LiveBench 目前包含 6 个类别中的 18 个不同任务，将随着时间的推移发布新的、更困难的任务。

网址:  
[https://livebench.ai/](https://livebench.ai/)

### 1.3. Fine-tuning Index (微调能上升的幅度) 
微调排行榜将受欢迎的在多个任务上进行微调的开源模型的性能与 GPT-4 进行了比较。

值得注意的是，大多数经过微调的开源模型都优于 GPT-4，其中 Llama-3、Phi-3 和 Zephyr 表现最佳。

网址:  
[https://predibase.com/fine-tuning-index](https://predibase.com/fine-tuning-index)

### 1.4. SuperCLUE
国内模型排行榜

Website:  
[https://www.superclueai.com/](https://www.superclueai.com/)

## 2. LLM组织和模型
| Organization | Product |OpenSource | Location |
| -- | -- | -- | -- |
| Foreign |||
| OpenAI | GPT | Close | US, UK（美国，英国） |
| Google | Gemini/Bard/Gemma/PaLM | Open |-|
| Anthropic | Claude | Close | US, UK（美国，英国） |
| Meta | **Llama**/Alpaca | Open |-|
| Microsoft | **Phi**/WizardLM/Bing | Open |-|
| Mistral | **Mistral/Mixtral** | Open | US, France（美国，法国） |
| HuggingFace | Zephyr | Open |-|
| Cohere | Command R | Open |-|
| NousResearch | Nous/OpenHermes | Open |-|
| LMSYS | Vicuna/FastChat |-|-|
| Reka AI | Reka | Open | US, UK, Singapore（美国，英国，新加坡） |
| Nvidia | Nemotron/NV/ChipNeMo | Open |-|
| Nexusflow | Starling | Open | Palo Alto, CA（美国加利福尼亚州门帕洛阿尔托） |
| Databricks/MosaicML | DBRX/Dolly/ MPT | Open | Many |
| OpenChat | OpenChat |-|-|
| Snowflake | Sonwflake | Close |-|
| UC Berkeley | Starling/Koala/Gorilla | Close |-|
| Perplexity AI | pplx | Close |-|
| Cognitive Computations | Dolphin | Open | Personal |
| Upstage AI | SOLAR | Open | 韩国 | 
| TII | falcon | Open | 阿拉伯 |
| Together AI | StripedHyena | Open | San Francisco（美国加利福尼亚州旧金山） |
| Allen AI | Tulu/OLMo | Open | Seattle, WA, United States（美国华盛顿州西雅图） |
| Nomic AI | GPT4All | Open | New York（美国纽约） |
| RWKV | RWKV | Open |-|
| OpenAssistant | OpenAssistant | Open |-|
| Stability AI | StableLM | Open | Canada（加拿大） |
| Bloomberg | BloombergGPT | Close | US, UK（美国，英国） |
| inflection.ai | Inflection | Close | San Francisco Bay Area（美国加利福尼亚州旧金山湾区） |
| xAI（Elon Mask） | Grōk | Close | San Francisco Bay Area, California, U.S（美国加利福尼亚州旧金山湾区）|
| Scale | Scale | Close | San Francisco（美国加利福尼亚州旧金山） |
| Character AI | Character | Close | Menlo Park, CA（美国加利福尼亚州门洛帕克） |
| Domestic |||
| Alibaba（阿里） | **Qwen（通义千问）** | Open | 杭州 |
| Tsinghua/Zhipu AI（清华&智谱AI） | **GLM/ChatGLM** | Open | 北京 |
| Baichuan（百川智能） | **Baichuan** | Open | 北京 |
| ModelBest（面壁智能） | **CPM** | Open | 北京 |
| 01 AI（零一万物） | Yi | Open | 北京 |
| DeepSeek AI（深度求索）| DeepSeek | Open | 杭州 |
| Colossal AI（潞晨科技） | Colossal | Open | 北京 |
| XVERSE（元象科技） | XVERSE | Open | 深圳 |
| Moonshot（月之暗面） | Moonshot | Close | 北京 |
| Step（阶跃星辰） | Step | Close | 上海 |
| MiniMax（稀宇科技） | ABAB | Close | 上海 |
| Baidu（百度） | ERNIE（文心一言） | Close | 北京 |
| SenseTime（商汤） | SenseChat（商量） | Close | 上海 |
| Bytedance（字节跳动） | Doubao/Coze | Close | 北京 |
| Tencent（腾讯） | Hunyuan（混元） | Close | 深圳 |
| 360 | 360gpt | Close | 北京 |






