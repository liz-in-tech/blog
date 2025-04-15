---
icon: lightbulb
sidebar: false
date: 2024-11-01
prev: ./013_optimizing_llm
next: ./011_vector_database
category:
  - LLM
tag:
  - Prompt
---
# 提示工程 Prompt Engineering
> 不管用什么 AI 工具，都是要把 LLM 用起来，核心点是写好 Prompt
  - 提示词的构成
  - 两大Prompt原则
  - 迭代优化Prompt
  - Prompt技术
<!-- more -->

## 1. 提示词的构成
- 指示（Instruction）: 描述要让模型做什么？
  - 人设：角色，行为，技能擅长点
  - 任务
  - 注意点：能做什么，不能做什么
- 上下文（Context）:
  - 给出与任务相关的背景信息
  - Retrieval: RAG场景中检索出的信息 
  - History: 多轮对话的历史对话 
- 例子（Examples）: 给出一些例子，让模型知道怎么回复
- 输入（Input）: 任务的输入信息 Query
- 输出（Output）: 输出的格式，想要什么形式的输出

## 2. 两大Prompt原则
### 2.1. 原则1：编写明确和具体的指令
- 编写明确和具体的指令，尽量不要让LLM去猜测，告诉它细节
  - 技巧1：使用分隔符或标题清楚地指示输入的不同部分
    - 使得指令清晰明了，避免指令冲突
    - 可以避免指令注入，如用户输入了“忘记之前指令”等内容
    - examples:
      - titles: ### 
      - Triple quotes:"""
      - Triple backticks:```
      - Triple dashes:---
      - Angle brackets: <>
      - xml tags:\<tag\>\</tag\>
  - 技巧2：要求结构化输出
    - 例如
      - html
      - json
      - markdown
  - 技巧3：Few-shot prompting
    - 提供成功执行任务的示例
    - 想要得到什么结果（验收标准）
  
![避免指令注入](../../../assets/012_avoid_prompt_injections.png)

### 2.2. 原则2：给模型时间去思考
- 给模型时间去思考
  - 技巧1：给出完成一个任务的每个具体步骤
    - example:
      - step 1:... step 2:... step N:...
  - 技巧2：引导模型将任务拆解，列出思考过程，一步步思考

## 3. 迭代优化Prompt

>好的Prompt需要不断调优

>没有一个适合所有场景的完美Prompt，要针对特定场景反复尝试优化

- 尝试：先写一份Prompt，看看会发生什么
- 分析：进行误差分析，找出没有达到预期效果的原因
  - 是否提示不够清晰
  - 是否没有给模型足够的时间思考
- 优化：逐步改进Prompt以更接近所需的结果，甚至可以改变解决问题的思路或方法
- 评估：根据大量案例进行评估
- 重复：反复循环多次，直到找到适合的Prompt

![迭代优化Prompt](../../../assets/012_iterative_prompt.png)

## 4. Prompt技术
### 4.1. xxx-shot：提供例子

通过少量例子，来调教llm

- zero-shot learning：不给任何例子
- one-shot learning：只给一个例子
- few-shot learning：给多个例子
  
思考：  
- 需要提供多少例子？ 提倡例子不要超过5-10个  
- 如果给了很多例子都不起作用怎么办？考虑微调，微调也可以看作是用很多例子来改变llm的行为

### 4.2. 思维链（Chain of Thoughts，CoT）
告诉大模型一步一步是怎么思考分析的

引导大模型一步步思考，把大问题拆解为一个个小问题
```
Let's think step by step.
```
```
让我们一步一步地思考
```

需要一定推理的问题，这个比较重要

### 4.3. Few-shot + CoT
对于需要推理的示例，也就是不是一眼就能看懂的，最好给出分析过程，多做一些引导，而不光只给出结果，可以提升点效果

### 4.4. 自我一致性
Few-shot + CoT + 多路径推理一致

采样多个不同的推理路径，并使用生成结果选择最一致的答案

### 4.5. 草稿链（Chain of Draft，CoD）
![CoD](../../../assets/012_cod.png)

CoD 在 CoT 基础上每个思考步骤不得超过 x 个 token，效果竟然很好，还节省成本

由 Zoom 研究团队提出，灵感来源于人类在解决复杂问题时仅记录关键信息的简洁思维模式，旨在提高推理效率并显著降低计算成本，与 CoT 相比，显著降低了 Token 使用量与延迟

Paper: https://arxiv.org/pdf/2502.18600

## 5. 积累
- Prompt中变化的部分用变量来表示
  - 用分隔符或标题将内容分为几个部分，有条理，使模型更容易理解
```
{instruction}

### 上下文
"""{context}"""

### 问题
"""{query}"""

### 输出格式
"""{output}"""

### 示例
"""{examples}"""
```

- 想要降低幻觉，可以在Instruction中给出这方面提示
```
如果不确定答案，请回答“不确定答案”。
```
- 用 “你” 来设定模型角色和行为
```
你是xxx专家，擅长xxx。

你在xxx方面有以下技能：
1.xxx
2.xxx
```
- 对话流程设定，不局限于一轮对话，例如模拟面试，设定接下来对话的一系列流程
```
### 开始对话前询问
1.要求用户提供简历
2.要求用户提供正在面试的职位
3.要求用户提供面试官的角色
4.询问用户想问多少问题 
5. ...
```

## 6. 优质Prompt
[LangChain Hub Prompt 参考：https://smith.langchain.com/hub](https://smith.langchain.com/hub)

## 7. 参考
https://www.promptingguide.ai/