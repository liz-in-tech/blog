---
icon: lightbulb
sidebar: false
date: 2024-02-21
prev: ./004_transformer
next: ./002_langchain_sourcecode
category:
  - LLM
tag:
  - Streamlit
---
# 使用Streamlit构建对话式应用程序
- 构建对话界面所需的组件
- 对话界面完整流程
- 流式输出assistant消息的改造
- 构建封装ChatGPT的应用
<!-- more -->

## 1. 构建对话界面所需的组件

- 1.两个聊天消息容器：分别显示来自用户和机器人的消息
- 2.聊天输入小部件：以便用户输入消息
- 3.存储聊天历史消息的列表：在每次用户或机器人发送消息时附加到该列表中。

### 1.1. st.chat_message 聊天消息容器，显示双方聊天消息

- 可以包含任何streamlit元素，包括图表、表格、文本等
- 要将元素添加到返回的容器中，可以使用with符号，也可以直接进行方法调用

### 1.2. st.chat_input 聊天输入组件，用户输入消息

- 返回值是用户的输入
- 可以传入默认提示显示在输入框内

### 1.3. st.session_state.messages 聊天历史消息列表

- 列表中每个条目都包含2个信息：角色role和消息内容content。
  
## 2. 对话界面完整流程
```python
import streamlit as st

st.title("Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input("Ask me anything!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = f"Echo: {prompt}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": "response"})
```

## 3. 流式输出assistant消息的改造
```python
# Streamd response emulator
def response_generator():
    # response example
    response = f"Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps. So let's get started!"

    for word in response.split():
        yield word + " "
        time.sleep(0.05)
```

```python
# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
```
## 4. 构建封装ChatGPT的应用
```python
from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
```