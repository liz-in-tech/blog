---
icon: lightbulb
sidebar: false
date: 2024-01-29
prev: ./003_streamlit
next: ./001_langchain
category:
  - LLM
tag:
  - LangChain
  - Source Code
---
# 从源码视角，窥探LangChain的运行逻辑
  - LangChain的基类
  - LCEL 与 Runnable
  - Chain 源码
  - AgentExecutor 源码
<!-- more -->

## 1. LangChain的基类
Python的抽象基类是ABC，LangChain的基类是其子类Runnable。

基类的继承关系如下图：

![基类的继承关系](../../../assets/002_base_class_inheritance_relationship.png)

RunnableSerializable的子类如下：

![RunnableSerializable的子类](../../../assets/002_subclass_of_runnableserializable.png)

抽取出重要的类，就有如下的继承关系：
- ABC
  - Runnable
    - Chain
      - AgentExecutor
    - BasePromptTemplate
    - BaseLanguageModel
    - BaseOutputParser    
    - BaseRetriever
    - BaseTool

## 2. LCEL 与 Runnable
直观理解：
- Runnable是一个零件
- LCEL是将零件组装为成品的方式
### 2.1. LCEL(LangChain的表达式语言) 
LCEL，是LangChain Expression Language的缩写，即LangChain的表达式语言。

LCEL是一种轻松地将Runnable组合为Chain的声明性方式。

#### 2.1.1. 组合Runnable的方式
##### 2.1.1.1. 串行(RunnableSequence)
- 特点
  - 顺序调用一系列Runnable
  - 上一个Runnable的输出作为下一个Runnable的输入
- 实现
  - 使用管道操作符"|"
  - 向RunnableSequence传递runnable列表
```python
from langchain_core.runnables import RunnableLambda

# 使用`|`操作符构建的RunnableSequence
sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
sequence.invoke(1) # 输出 4
sequence.batch([1, 2, 3]) # 输出 [4, 6, 8]
```
##### 2.1.1.2. 并行(RunnableParallel)
- 特点
  - 并行Runnable有同一个输入
  - 并行Runnable有各自的输出
- 实现
  - 使用dict
  - 向RunnableParallel传递dict
```python
from langchain_core.runnables import RunnableLambda

# 包含用字典字面量构建的RunnableParallel的序列
sequence = RunnableLambda(lambda x: x + 1) | {
    'mul_2': RunnableLambda(lambda x: x * 2),
    'mul_5': RunnableLambda(lambda x: x * 5)
}
sequence.invoke(1) # 输出 {'mul_2': 4, 'mul_5': 10}
```
### 2.2. Runnable
#### 2.2.1. What
Runnable是一个可以被调用、批处理、流式处理、转换和组合的工作单元，也就是可以组合成链(Chain)的零件。
#### 2.2.2. 功能
- 支持同步和异步
- 支持批处理
- 支持流式处理
#### 2.2.3. 关键方法
- invoke/ainvoke：一个输入=>一个输出
- batch/abatch：多个输入=>多个输出
- stream/astream：一个输入=>流式输出
- astream_log：一个输入=>流式输出和选定的中间结果
### 2.3. Runnable的子类
- Chain
  - AgentExecutor
- BasePromptTemplate
- BaseLanguageModel
  - BaseLLM
  - BaseChatModel
- BaseOutputParser    
- BaseRetriever
- BaseTool
### 2.4. Runnable子类的输入和输出类型
#### 2.4.1. Chain
- invoke
  - input:Dict[str, Any]
  - output:Dict[str, Any]
- \_\_call\_\_
  - input:Union[Dict[str, Any], Any]
  - output:Dict[str, Any]
- run
  - input:Any
  - output:Any
#### 2.4.2. BasePromptTemplate
- invoke
  - input:Dict
  - output:PromptValue
#### 2.4.3. BaseLLM
- invoke
  - input:Union[PromptValue, str, Sequence[BaseMessage]]
  - output:str
#### 2.4.4. BaseChatModel
- invoke
  - input:Union[PromptValue, str, Sequence[BaseMessage]]
  - output:BaseMessage
#### 2.4.5. BaseOutputParser
- invoke
  - input:Union[str, BaseMessage]
  - output:T
#### 2.4.6. BaseRetriever
- invoke
  - input:str
  - output:List[Document]
#### 2.4.7. BaseTool
- invoke
  - input:Union[str, Dict]
  - output:Any
## 3. Chain
### 3.1. What
- 抽象基类
- 作用：构造组件的调用序列
- 特点
  - 有状态（Stateful）：为任何 Chain 添加 Memory，即可为其赋予状态。
  - 可观察（Observable）：为任何 Chain 添加 Callbacks，即可在组件调用序列外附加额外的功能，如日志记录。
  - 可组合（Composable）：Chain API 足够灵活，便于将 Chain 与其他组件结合，包括其他 Chain。
### 3.2. Chain的属性
- Memory：在chain开始的时候组装inputs，在chain结束的时候保存inputs和outputs。
- Callbacks：在Chain的调用生命周期中，从on_chain_start开始，以on_chain_end或on_chain_error结束，都会调用Callbacks处理程序。
### 3.3. Chain的源码解读
源码入口：Chain的invoke方法

源码位置：langchain.chains.base.py中
```python
def invoke(
    self,
    input: Dict[str, Any],
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    config = ensure_config(config)
    callbacks = config.get("callbacks")
    tags = config.get("tags")
    metadata = config.get("metadata")
    run_name = config.get("run_name")
    include_run_info = kwargs.get("include_run_info", False)
    return_only_outputs = kwargs.get("return_only_outputs", False)

    inputs = self.prep_inputs(input) # 验证和准备chain的输入，包括添加来自memory的输入
    callback_manager = CallbackManager.configure(
        callbacks,
        self.callbacks,
        self.verbose,
        tags,
        self.tags,
        metadata,
        self.metadata,
    )
    new_arg_supported = inspect.signature(self._call).parameters.get("run_manager") # 看是否支持run_manager，即看_call的参数是否有run_manager
    run_manager = callback_manager.on_chain_start(
        dumpd(self),
        inputs,
        name=run_name,
    )
    try:
        outputs = (
            self._call(inputs, run_manager=run_manager) # chain的子类实现
            if new_arg_supported
            else self._call(inputs)
        )
    except BaseException as e:
        run_manager.on_chain_error(e)
        raise e
    run_manager.on_chain_end(outputs)
    final_outputs: Dict[str, Any] = self.prep_outputs( # 验证和准备chain的输出，并将此次运行的信息保存到memory中
        inputs, outputs, return_only_outputs
    )
    if include_run_info:
        final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
    return final_outputs
```
invoke源码逻辑：
- prep_inputs: 验证和准备chain的输入，包括添加来自memory的输入
  - memory.load_memory_variables: memory相关
- callback.on_chain_start: callback相关
- **self._call: 
	抽象方法,由Chain的子类实现**
- callback.on_chain_error: callback相关
- callback.on_chain_end: callback相关
- prep_outputs: 验证和准备chain的输出，并将此次运行的信息保存到memory中
  - memory.save_context: memory相关，将inputs, outputs保存到memory
## 4. AgentExecutor
### 4.1. What
- Chain的子类
- 可以使用工具、拥有记忆、动态决策的智能体
- AgentExecutor = Agent + Memory + Tools
- 对用户的问题，进行多轮迭代执行，最终返回结果
### 4.2. AgentExecutor的属性
- Memory：继承自父类Chain
- Callbacks：继承自父类Chain
- Agent：决定迭代执行的每一步的行为
- Tools：可以使用的工具
- Max Iterations：最大迭代轮次
- Max Execution Time：最大执行时间
### 4.3. AgentExecutor的源码解读
#### Step1：Chain.invoke
AgentExecutor没有重写invoke方法，会调用父类的invoke方法。
#### Step2：AgentExecutor._call(迭代器，循环迭代执行)
```python
    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)
```
_call源码逻辑：
- while _should_continue：迭代器
  - _take_next_step：迭代方法
#### Step3：迭代停止条件
- 迭代次数限制：iterations>=self.max_iterations
- 迭代时间限制：time_elapsed>=self.max_execution_time
- 停止信号AgentFinish：isinstance(next_step_output, AgentFinish)
#### Step4：_take_next_step(迭代方法)
```python
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        return self._consume_next_step(
            [
                a
                for a in self._iter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                )
            ]
        )
```
_take_next_step源码逻辑：
- _iter_next_step
  - agent.plan
  - tool.run
- _consume_next_step
#### Step5：_iter_next_step
_iter_next_step源码逻辑：
- 1.调LLM生成一个计划plan（plan要么是AgentFinish，要么是AgentAction）
- 2.如果是AgentFinish则返回，结束
- 3.如果是AgentAction则调对应的工具
- 4.调对应工具生成observation
- 5.生成并返回AgentStep(action=agent_action, observation=observation)，结束
#### Step6：_consume_next_step
```python
def _consume_next_step(
    self, values: NextStepOutput
) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
    if isinstance(values[-1], AgentFinish):
        assert len(values) == 1
        return values[-1]
    else:
        return [
            (a.action, a.observation) for a in values if isinstance(a, AgentStep)
        ]
```
_consume_next_step源码逻辑：
- 如果是AgentFinish，直接返回
- 如果是AgentStep，则转换为List[Tuple[AgentAction,str]]后返回
### 4.4. AgentExecutor的源码总结
![AgentExecutor源码](../../../assets/002_agentexecutor_sourcecode.jpg)

更多细节：

![AgentExecutor源码更多细节](../../../assets/002_agentexecutor_sourcecode_detail.png)