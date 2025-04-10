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
# Building Conversational Applications with Streamlit
- Components Required for Constructing a Dialogue Interface
- Complete Process of Dialogue Interface
- Refactoring for Streaming Output of Assistant Messages
- Building an Application that Encapsulates ChatGPT
<!-- more -->

## 1. Components Required for Constructing a Dialogue Interface

- 1. Two chat message containers: to display messages from the user and the robot respectively.
- 2. Chat input widget: for the user to input messages.
- 3. A list to store chat history messages: appended to this list each time a message is sent by the user or the robot.

### 1.1. st.chat_message Chat message container, displays chat messages from both parties

- Can include any Streamlit elements, including charts, tables, text, etc.
- To add elements to the returned container, you can use the `with` statement or call methods directly.

### 1.2. st.chat_input Chat input component, user inputs messages

- The return value is the user's input.
- A default prompt can be passed to display in the input box.

### 1.3. st.session_state.messages Chat history message list

- Each entry in the list contains two pieces of information: role and message content.

## 2. Complete Process of Dialogue Interface
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
    st.session_state.messages..append({"role": "assistant", "content": "response"})
```

## 3. Refactoring for Streaming Output of Assistant Messages
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
## 4. Building an Application that Encapsulates ChatGPT
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