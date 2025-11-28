---
icon: lightbulb
sidebar: false
date: 2025-11-27
prev: false
next: ./040_mem0
category:
  - LLM
tag:
  - CrewAI
  - Agent
  - Multi-Agent
---
# CrewAI: 主流的多智能体框架

- CrewAI的两种自动化方式：Crews & Flows
- CrewAI Flows的流程控制逻辑
- CrewAI Crews的三大模块化组件（Agents，Tasks，Crews）
- CrewAI使用案例：股票分析系统

<!-- more -->

## 相关概念梳理
Traditional Automation vs Agentic Automation
- Traditional Automation：依靠条件判断实现，有明确的规则驱动，固定可复现，适用于确定性、重复性任务
- Agentic Automation：借助LLM的模糊推理能力，通过语义理解和上下文推理来驱动，自主决策，多路径可选，动态多样，适应性强，适用于模糊性、高语义任务（如内容生成、决策分析）

Agent vs Multi-Agent
- Agent
    - Memory: 持续追踪上下文逻辑与全局目标
    - Tool: 与现实世界互动，执行实际操作
        - 调用外部 API 和服务
        - 集成企业内部系统（CRM、数据库等）
        - 执行复杂计算或处理任务
        - 与现有程序进行通信和交互
    - 是一个ReAct循环: Observation -> Thinking -> Action
- Multi-Agent
    - 高度专业化：每个Agent只聚焦特定任务
        - 每个智能体应只承担特定任务范围，控制上下文长度与工具数量
            - 工具应精挑细选，与任务高度匹配。过多工具会导致模型选择困难或误用，尤其影响小模型表现。
    - 多智能体协作：模拟人类团队的交流与反馈，智能体间应能交流信息、互相评审、委派任务，形成反馈闭环，提升结果质量。
    - 模型灵活性：不同Agent可以针对任务特点采用不同模型
    - 整体结果的质量提升：相较于单一Agent执行多任务，多个高专注度的Agent协作往往能生成更精确、更专业的结果
    - 更进一步：Self-Improvement Loop: 自我改进机制，通过反馈不断优化决策逻辑与执行结果
        - 智能体通过记忆学习与优化，随着时间推移输出更加可靠

## CrewAI的两种自动化方式：Crews & Flows

![](../../../assets/041_crewai.png)

- 从宏观层面来看，CrewAI 创建了两种主要的自动化方式：团队Crews & 流程Flows
    - Crew：自主协作的AI团队，其中每个智能体都有特定的角色、工具和目标（动态的智能决策）
        - Crew: 团队
        - Agents: 智能体
        - Tasks: 任务
        - Process: 协作模式（顺序执行 Sequential or 层级式执行 Hierarchical）
    - Flow：实现精细的、事件驱动的控制，精确的任务编排，并原生支持 Crews（静态的工作流）
        - Flow: 流程，工作流编排
        - Events: 事件
        - States: 状态
        - Crew Support: 原生支持Crews
- When to Use Crews vs. Flows
    - 使用Crews场景
        - 开放性问题，探索性任务，需要自主解决问题，创造性协作
    - 使用Flows场景
        - 可预测、可审计、具有确切路径
    - 结合Crews和Flows场景
        - 既需要结构化的流程，也需要一些自主智能

## CrewAI Flows的流程控制逻辑
- @start()：流程的起点
- @listen()：监听器（当被监听的方法产生输出时，监听器方法会被触发）
    - 条件逻辑 or
        - 用于监听多个方法，并在任何指定方法发出输出时触发监听器方法
    - 条件逻辑 and
        - 仅当所有指定的方法都发出输出时才触发监听器方法
- @router()：根据方法的输出指定不同的路由
- @persist：管理状态的持久化，可应用于类级别和方法级别

## CrewAI Crews的三大模块化组件（Agents，Tasks，Crews）
通过这三大组件，CrewAI 实现了从任务定义、角色分配到结果输出的完整智能工作流。

- Agents（智能体）：具备目标导向行为的智能体
    - 功能描述：多智能体系统的核心个体，具备自主思考与行动能力。
    - 关键属性
        - Role（角色）：定义职责与身份
        - Goal（目标）：明确任务方向
        - Backstory（背景故事）：提供行为语境与决策依据
- Tasks（任务）：智能体执行的具体目标或操作单元
    - 功能描述：指导智能体执行的具体工作内容。前一任务输出可作为后一任务的输入。
    - 关键属性
        - Description（任务描述）：明确告知智能体任务内容是什么
        - Expected Output（预期结果）：定义结果形式与质量标准，明确告知智能体期望完成的结果，确保输出与目标一致
            - Expected Output 是一种 “强制函数（forcing function）”，它引导智能体思考如何达成目标，并确保输出符合结构化要求。这一机制能有效提升输出的可控性与一致性。
        - Agent（关联智能体）：指定执行者
- Crews（团队） /kruː/：一个 Crew 是由多个具备不同角色与能力的 AI 智能体组成的团队，通过协作完成复杂任务。
    - 功能描述：将智能体与任务组合为可运行单元，实现协同运作。
    - 关键属性：agents（智能体列表）、tasks（任务列表）、verbose（详细模式，设置 verbose=2 可启用最高级别日志输出）

## CrewAI Agents协作模式
- 传统协作方式 (Sequential & Parallel)
    - 顺序执行 (Sequential)
        - 局限性：初始上下文可能随任务链拉长而丢失
    - 并行执行 (Parallel Execution)
        - 局限性：仅解决任务执行速度问题，不解决上下文丢失
- 进阶协作方式：层级式调度 (Hierarchical Process)
    - 核心优势：
        - 单点控制：由经理智能体 (Manager Agent) 统一管理。
        - 记忆目标：经理保持初始目标不丢失。
        - 自动委派：经理将任务分配给其他智能体。
        - 结果审查：经理审查并可要求改进任务输出。
    - 实现方式：在 CrewAI 中，仅需少量代码即可切换至层级流程。
    - 用户自定义：可以自定义经理智能体的策略与决策方式。

## CrewAI Tools
CrewAI提供了丰富的内置工具与技能库，并支持外部工具和自定义工具
- 内置工具：提供多种开箱即用工具。
- LangChain 兼容性：CrewAI 支持所有 LangChain 工具，极大扩展了智能体可使用的工具集。
- 自定义工具：开发者可以创建自定义工具，与外部 API、数据库、CRM 或内部系统无缝集成。

CrewAI Tools的两种工具作用域（Tool Scope）
- Task 级
    - 工具仅在该任务执行时可用，优先级高
- Agent 级
    - 工具在Agent的所有任务中可用，优先级低

## CrewAI的三类记忆机制
只需在 Crew 实例化时开启一个标志即可启用记忆功能，框架自动管理存储与调用。

启用记忆后，CrewAI 会自动激活以下三类机制，分别服务于不同层级的学习与上下文保持。
- 短期记忆（Short-Term Memory）：
    - 存储单次客户对话中的上下文与中间结果。
    - 生命周期：仅在当前 Crew 执行期间存在。
    - 功能：存储执行任务过程中产生的信息与中间结果。
    - 特性：在同一 Crew 内，所有智能体共享短期记忆，实现上下文连贯与多体协作。
    - 用途示例：在“规划-撰写-编辑”流程中，编辑可直接访问规划师生成的大纲与撰稿人的初稿。
- 长期记忆（Long-Term Memory）：
    - 保留过往客户交互与自我批评结果，帮助系统不断改进。
    - 生命周期：Crew 执行结束后仍保留，存储于本地数据库。
    - 功能：支持智能体从历史执行中“学习经验”。
    - 机制：
        - 执行完成后，智能体会进行 自我批评（Self-Critique）；
        - 记录如何改进下一次表现；
        - 这些反馈被保存并在未来运行时自动调用。
    - 效果： 让智能体“越用越聪明”，持续优化输出质量与一致性。
- 实体记忆（Entity Memory）：
    - 追踪客户、公司、产品等核心“实体”，以实现跨任务引用与个性化响应。
    - 生命周期：与短期记忆相似，仅在当前执行期间有效。
    - 功能：存储智能体正在处理的主题或实体信息。
    - 示例：
        - 当研究智能体分析 “OpenAI 公司” 时，系统会创建 “OpenAI” 实体；
        - 其中保存了相关的产品、人员、历史与特征，供其他任务引用。

## CrewAI任务的高级超参数
高级超参数用于处理复杂任务场景，可让复杂多智能体系统更灵活、安全且可控。

高级属性允许任务实现 结构化输出、人工审查与并行处理，提高系统灵活性和效率。

- 上下文 (Context)	
    - 作用：为任务提供额外背景信息。	
    - 示例：在任务开始前加载最新数据或指令。
- 回调函数 (Callback)	
    - 作用：任务完成后执行自定义函数。	
    - 示例：发送通知邮件或记录数据。
- 工具覆盖 (Overriding Agent Tools)	
    - 作用：用任务特定工具覆盖智能体默认工具。	
    - 示例：限制任务只能使用指定工具，提高安全性与效率。
- 强制人类输入 (Force Human Input)	
    - 作用：完成前暂停任务，等待人工审查。	
    - 示例：在关键决策点进行人工干预。
        - 可在关键节点暂停并等待人工审查或输入，确保重要决策质量。
- 执行模式 (Execution)	
    - 作用：配置任务的运行方式。	
    - 示例：同步 (synchronous) 或 并行 (parallel) 执行。
        - 并行执行 (Parallel Execution)：允许任务异步运行，提升整体系统效率。
- 输出格式 (Output Format)	
    - 作用：指定任务输出的格式，便于后续编程、存储或作为函数参数使用。	
    - 示例：输出为 Pydantic 对象、JSON 或文件。
        - 使用 Pydantic 模型将 LLM 模糊输出转换为强类型数据对象。
        - 可输出为 JSON 文件 或其他文件格式，便于与常规编程系统集成。

## 智能体的设计
- 智能体设计的管理者思维
    - 设计智能体时，应从“我能用 AI 做什么？”转向：“如果我要雇佣一群人来完成这个任务，我会雇佣谁？”
    - 强调角色定位、任务分工、目标明确是多智能体系统成功的关键
- 精确角色定位与关键词的力量
    - 仅给智能体一个通用角色（如“研究员”或“作家”）容易产生模糊、低质量的输出。优秀智能体需要 精确的角色定义和 关键词 来引导行为，就像雇佣领域专家一样。
    - 在定义智能体的角色、目标和背景故事时，务必使用 最能代表专业知识的关键词。
    - 示例
        - 研究员 (Researcher) -> HR 研究专家 (HR Research Specialist)
            - 限制研究范围，输出更专业的 HR 知识
        - 作家 (Writer) -> 高级文案 (Senior Copywriter)
            - 提高输出质量、语气及营销专业性
        - 金融分析师 (Financial Analyst) -> FINRA 批准分析师 (FINRA Approved Analyst)
            - 使用关键词（如 FINRA）引导 LLM，触发 RAG 机制，输出符合行业标准的分析结果
- 创建出色任务的强制属性
    - 清晰描述 (Clear Description)：明确告知智能体任务内容是什么
    - 清晰的预期结果 (Clear Expected Output)：明确告知智能体期望完成的结果，确保输出与目标一致
- 构建出色工具的三大关键要素
    - 多功能性 / 通用性（Versatility）
        - 工具是 AI 应用模糊输入与外部系统强类型输入之间的转换器。
        - 实现要点：工具必须足够灵活，能处理 LLM 提供的多样化输入，提高复用性
        - 开发实践：确保 LLM 输出的参数可正确转换为外部 API 或服务所需的类型，以保证请求执行成功。
    - 容错性（Fault-Tolerance）
        - 背景：外部系统可能发生异常，影响智能体流程。
        - 作用：工具必须能够优雅失败（Fail Gracefully）并具备自愈能力。保证系统即使遇到错误仍能继续执行，提升鲁棒性。（Self-healing），防止流程中断。
        - CrewAI 机制：
            - 即使工具运行异常，Crew 不会停止执行。
            - 智能体会根据错误信息采取行动，例如调整输入、补充参数或尝试其他执行方法。
        - 大规模部署时考虑：工具需能处理复杂文档、不符合预期的数据格式，确保系统稳定运行。
    - 智能缓存（Caching）
        - 背景：工具调用外部 API 或服务时可能触发速率限制或消耗时间。
        - 作用：缓存机制可优化系统性能，减少重复调用，提高多智能体系统效率。节省 API 调用成本，避免速率限制，加快系统运行。
        - CrewAI 特性：跨智能体缓存
            - 若不同智能体使用相同工具和参数调用请求，系统会命中缓存，避免重复调用。
            - 减少 API 请求次数，避免速率限制，并加速系统执行。


## 使用CrewAI创建多智能体系统的步骤
- 定义LLM（三种方式：环境变量，使用 YAML 配置或直接在代码中定义）
- 定义Agents（两种方式：使用 YAML 配置或直接在代码中定义）
- 定义Tasks（两种方式：使用 YAML 配置或直接在代码中定义）
- 定义Tools
- 创建团队Crew
- 运行Crew

## 使用案例：股票分析系统

三个团队Crews
- 数据收集团队 (DataCollectionCrew) Process.sequential, allow_delegation=False
    - 市场研究员 - 市场研究任务: 收集市场新闻、行业信息和公司动态
        - market_researcher -> market_research_task
    - 财务数据专家 - 财务数据收集任务: 获取财务报表、关键财务指标
        - financial_data_expert -> financial_data_collection_task
    - 技术分析师 - 技术分析任务: 收集价格数据、技术指标
        - technical_analyst -> technical_analysis_task
    - 数据验证专家 - 数据验证任务: 验证数据质量、处理异常值
        - data_validation_expert -> data_validation_task
    - 数据协调专家 - 数据协调任务: 协调和整合所有数据收集工作
        - data_coordination_agent -> data_coordination_task
- 分析团队 (AnalysisCrew) Process.hierarchical, allow_delegation=True
    - 高级基本面分析师 - 基本面分析任务: 评估公司基本面、财务健康状况
        - fundamental_analyst -> fundamental_analysis_task
    - 风险评估专家 - 风险评估任务: 分析投资风险、风险因素识别
        - risk_assessment_specialist -> risk_assessment_task
    - 行业研究专家 - 行业分析任务: 分析行业地位、竞争环境
        - industry_expert -> industry_analysis_task
    - 量化分析师 - 量化验证任务: 验证分析结论的统计显著性
        - quantitative_analyst -> quantitative_validation_task
    - 分析协调员 - 分析协调任务: 整合所有分析结果并形成最终判断
        - analysis_coordinator -> analysis_coordination_task
- 决策团队 (DecisionCrew) Process.hierarchical
    - 投资策略顾问 - 投资策略制定任务: 生成投资建议、策略制定
        - investment_advisor -> investment_strategy_task
    - 报告生成专家 - 报告生成任务: 生成详细分析报告
        - report_generator -> report_generation_task
    - 质量控制专家 - 质量保证任务: 质量控制、结果验证
        - quality_assurance_specialist -> quality_assurance_task
    - 风险管理专家 - 风险评估任务: 从风险管理角度评估投资建议
        - risk_manager -> risk_assessment_task
    - 投资组合经理 - 投资组合优化任务: 从投资组合角度评估投资决策
        - portfolio_manager -> portfolio_optimization_task
    - 市场策略师 - 市场时机分析任务: 分析投资时机和策略
        - market_strategist -> market_timing_task
    - 道德合规官 - 合规审查任务: 从道德和合规角度审查投资建议
        - ethics_compliance_officer -> compliance_review_task
    - 决策主持人 - 集体决策任务: 主持投资决策委员会，进行集体讨论和决策
        - decision_moderator -> collective_decision_task

智能投资流程Flow
- 初始化分析 initialize_analysis 
    - @start()
- 智能路由数据收集策略 route_data_collection
    - @listen("initialize_analysis")
    - @router
        - 标准数据收集 standard_data_collection
            - @listen("route_data_collection")
        - 全面数据收集 comprehensive_data_collection
            - @listen("route_data_collection")
        - 实时数据收集 real_time_data_collection
            - @listen("route_data_collection")
    - DataCollectionCrew.execute_data_collection
- 智能路由分析策略 route_analysis_strategy
    - @listen(or_("standard_data_collection", "comprehensive_data_collection", "real_time_data_collection"))
    - @router
        - 深度分析 deep_analysis
            - @listen("route_analysis_strategy")
        - 标准分析 standard_analysis
            - @listen("route_analysis_strategy")
        - 快速分析 rapid_analysis
            - @listen("route_analysis_strategy")
        - 简化分析 simplified_analysis
            - @listen("route_analysis_strategy")
    - AnalysisCrew.execute_collaborative_analysis
- 智能路由决策策略 route_decision_strategy
    - @listen(or_("deep_analysis", "standard_analysis", "rapid_analysis", "simplified_analysis"))
    - @router
        - 集体决策 collective_decision
            - @listen("route_decision_strategy")
        - 标准决策 standard_decision
            - @listen("route_decision_strategy")
        - 快速决策 rapid_decision
            - @listen("route_decision_strategy")
        - 保守决策 conservative_decision
            - @listen("route_decision_strategy")
    - DecisionCrew.execute_collective_decision
- 完成分析并生成总结报告 finalize_analysis
    - @listen(or_("collective_decision", "standard_decision", "rapid_decision", "conservative_decision"))
- OUTPUT: 个性化分析结果

参考Github: https://github.com/liangdabiao/crewai_stock_analysis_system


