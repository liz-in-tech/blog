---
icon: lightbulb
sidebar: false
date: 2024-10-30
prev: ./008_rag_challenge
next: ./006_llm_leaderboard
category:
  - LLM
tag:
  - Claude
  - Computer Use
---
# Claude 3.5 Sonnet: Computer use
  - Industry Benchmarks 
  - API Support
  - Features of Computer Use
  - Future directions 
<!-- more -->

## 1. About
Company：Anthropic

Official Articles：

[1. Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku](https://www.anthropic.com/news/3-5-models-and-computer-use)

[2. Developing a computer use model](https://www.anthropic.com/news/developing-computer-use)

Official Demo Videos：

https://www.youtube.com/playlist?list=PLf2m23nhTg1NcDu3_eZavbTJ3Dow9QQRB

Code Implementation：

https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo

## 2. Industry Benchmarks
-  SWE-bench Verified（coding）
-  TAU-bench（tool use）
-  OSWorld (computer control)
  
Claude 3.5 Performance Evaluation Results：

For OSWorld, Claude 3.5 Sonnet scored 14.9% in the screenshot-only category.

![Claude3.5 Performance Evaluation Results](../../assets/007_claude_evaluation.png)

## 3. API Support
Support for calls via Anthropic API, Amazon Bedrock, and Google Cloud’s Vertex AI

- Anthropic API
- Amazon Bedrock
- Google Cloud’s Vertex AI

## 4. Features of Computer Use 
Use computers the way people do — by looking at a screen, moving a cursor, clicking buttons, and typing text.

- 1. translate instructions into computer commands (e.g.    fill out a form with the data from those pages; and so on)
  -  user commands
     -  use data from my computer and online to fill out this form
  -  translate into a series of computer commands
     -  check a spreadsheet;
     -  move the cursor to open a web browser;
     -  navigate to the relevant web pages;
     -  fill out a form with the data from those pages.
- 2. to interpret what’s happening on a screen and then use the software tools available to carry out tasks
- 3. Position the cursor to a specific location on the screen
  - calculate how many pixels the cursor needs to move vertically or horizontally to click on the right spot
  - Training Claude to count pixels accurately was critical. 
  - Claude trained on this.
- 4. self-correct and retry tasks when it encountered obstacles
 
## 5. Future directions
- Improve safety
- Reduce costs
- Increase speed
- Reduce Error Rate and increase Reliability
- Support more operations
  - Operations such as drag and drop, zoom, etc. are not yet supported.
- Performance Improvement of Visual Large Language Models (VLMs)
  - Current VLMs are generally ‘myopic’ in the sense that they can only extract approximate and abstract visual information from an image, and do not really see the picture as a human would.
- Changing viewing screenshots to finer-grained video streams
  - Claude observes the screen by taking screenshots that are ‘flip-book’ style, i.e. taking screenshots and piecing them together, rather than observing a finer-grained video stream, which means that it may miss brief actions or notifications.
- Combination of visual and code manipulation
  - Relying on vision alone doesn't improve speed or accuracy enough


