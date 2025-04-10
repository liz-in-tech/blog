---
icon: lightbulb
sidebar: false
date: 2024-10-31
prev: ./010_rag_workflow
next: ./008_rag_challenge
category:
  - LLM
tag:
  - LLM
  - RAG
  - Challenge
---
# Challenges in the Commercialization of Large Language Models
  - Current Solutions for Rapid Commercial Deployment: RAG
  - Challenges in the Commercialization of LLM
  - Generation and Retrieval
  - Use Case: Implementation of an Intelligent Customer Service System
<!-- more -->

## 1. Current Solutions for Rapid Commercial Deployment: RAG
- RAG (Retrieval-Augmented Generation)
- RAG is a practical approach for achieving rapid commercialization at the current stage
- It enables quick integration of existing business data into the LLM domain

## 2. Challenges in the Commercialization of LLM
### 2.1. Performance
- The requirements differ between B2B and B2C
  - B2B: Enterprises have high expectations for performance and accuracy
  - B2C: Generally lower expectations with a higher tolerance for errors, as users are more forgiving

### 2.2. Controllable Generation
- LLMs generate content with a certain level of freedom, which means the output is not always as intended
- Implementing controls or constraints can help guide model output to align with desired outcomes

#### 2.3. Privacy
- Models like GPT-4 can potentially extract user privacy information through guided prompts

#### 2.4. Hallucination
- Models sometimes generate plausible-sounding but inaccurate information ("hallucinations", /həˈluː.sɪ.neɪt/)
  - Makes statements that sound plausible but are not true
  - Generate fictitious information such as invalid URLs or non-existent numbers
- How to reduce hallucinations
  - First find relevant information, then answer the question based on the relevant information.
- Question: Can relying solely on the provided context completely resolve the hallucination problem in models?
  - It cannot completely resolve it, but it can significantly reduce hallucinations
  - Reasons it cannot be fully resolved:
    - Retrieval accuracy
    - LLM comprehension capabilities
    - Complexity of the question

## 3. Generation and Retrieval
### 3.1. Generation
- Pros: Content diversity and creativity
- Cons: Lack of control over the generated content

### 3.2. Retrieval
- Pros: Provides controlled and reliable output
- Cons: Limited by the boundaries of available content

### 3.3. Combining Generation and Retrieval: RAG
- RAG, or Retrieval-Augmented Generation, combines retrieval to support and enhance generation
- It brings together the strengths of both generation and retrieval

## 4. Use Case: Implementation of an Intelligent Customer Service System
### 4.1. Retrieval-Based Approach
- 1. Build a set of frequently asked question pairs (FAQ, i.e., <Q, A>)
- 2. Retrieve related questions based on the user’s query
- 3. Provide the answer to the most relevant question as the result

Characteristics:
- Requires the construction of a Q&A dataset
- Provides accurate and reliable answers

### 4.2. Generation-Based Approach
- 1. Build a set of Q&A pairs
- 2. Train the model on this Q&A dataset
- 3. The model generates an answer based on the user’s query

Characteristics:
- Requires constructing a Q&A dataset
- The generated responses are less reliable and harder to control

### 4.3. Combined Retrieval and Generation Approach
- 1. Build a knowledge base (not necessarily in Q&A format)
- 2. User inputs a question
- 3. Retrieve relevant information from the knowledge base as candidates
- 4. Input the user’s question, the candidate information, and previous conversation history into a prompt
- 5. Feed the prompt into the large language model (LLM), which generates the final response

Characteristics:
- No need to organize data into a Q&A format
- The LLM's generated response is based on information retrieved from the knowledge base, leading to more reliable results
