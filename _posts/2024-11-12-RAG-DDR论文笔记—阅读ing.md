---
layout: article
title: RAG-DDR论文笔记
key: post66
mode: immersive
tags:
  - nlp
  - 自然语言处理
  - 论文笔记
header:
  theme: ocean
article_header:
  type: overlay
  theme: ocean
  background_color: "#f1f8ff"
  background_image: false
excerpt_separator: <!---more-->
---
# RAG-DDR: OPTIMIZING RETRIEVAL-AUGMENTED  GENERATION USING DIFFERENTIABLE DATA REWARDS 论文笔记

<!---more-->

> 论文来源: 师兄给的
>  



## 不懂的问题

### 什么是 agent？

文中的“agent”指的是 RAG 系统中负责特定任务的独立模块或组件。在 RAG-DDR 框架中，agent 通常指的是系统中用于检索和生成的两个模块，分别为“检索代理（Retriever Agent）”和“生成代理（Generator Agent）”。

- **检索代理（Retriever Agent）**：负责根据用户查询从外部知识库或文档集合中检索相关信息。这个模块的任务是找到能够为生成模块提供有用背景知识的文档或数据。

- **生成代理（Generator Agent）**：接收检索到的信息，并基于这些内容生成最终的输出（如回答或响应）。生成代理的主要目标是综合这些信息生成符合用户需求的文本。

在 DDR 方法中，两个 agent 之间相互配合，通过数据偏好的对齐来提升整体性能。DDR 通过奖励机制，让每个 agent 在决策时不仅考虑自身的需求，还要确保生成的输出在系统层面上能为另一个 agent（即对方模块）提供更有利的信息支持。这种协同优化使得系统的检索和生成流程更加协调，提升了生成内容的准确性和一致性。

### 什么是agent的数据偏好？

文中提到的“数据偏好”指的是 RAG 系统中不同模块（如检索模块和生成模块）对数据需求的特定倾向。简单来说，不同模块在处理任务时所需要的信息可能存在差异，因此它们在数据选择和使用上具有不同的“偏好”。

在检索增强生成系统中，检索模块的主要任务是从外部资源中找到与用户查询相关的文档或信息，但这些检索到的内容并不一定都适合生成模块。例如，生成模块通常需要的是能够直接回答用户问题的核心信息，而不是一堆冗余或含糊的信息。因此，如果检索模块提供的信息不符合生成模块的需求，可能会导致生成质量下降，出现内容不准确、混淆或所谓的“幻觉”现象。

DDR 方法通过奖励机制，使得系统能够在训练过程中调整检索和生成模块的数据偏好，逐渐对齐它们的需求。通过这种数据偏好对齐，检索模块可以更有针对性地提供生成模块需要的核心信息，而生成模块也能够更有效地利用这些检索内容，从而提升系统整体的生成效果。


## 摘要

> 什么是agent？
> 什么是agent的数据偏好？

### 问题

SFT使得LLMs处理不同指导下的各种RAG任务。然而，它训练RAG模块过度拟合训练信号，并忽略了RAG系统内各个代理之间不同的数据偏好。
### 解决办法

Differentiable Data Reward method (DDR) 通过调整不同RAG模块之间的数据偏好，从而端到端地训练RAG系统。DDR通过使用回滚方法收集奖励来优化每个代理。

1. DDR 采用一种回滚方法，让每个代理对响应进行采样并加入一些细微变化（扰动），然后通过这些响应对整体系统性能的影响来评估其质量。该奖励信号促使代理生成更有助于提升系统最终输出质量的响应。

## 实验结果

1. DDR方法在性能上明显优于SFT方法，特别是对于更依赖于检索知识的参数较小规模的LLMs。
2. DDR方法还展现出更强的能力来调整RAG模块之间的数据偏好。
3. DDR方法使得生成模块更有效地从文档中提取关键信息，并缓解了参数化记忆与外部知识之间的冲突

##  引言

### LLM现状

1. LLM 的广泛应用
2. 优势：LLM 由于幻觉问题，通常会产生错误的响应，因此采用了检索增强生成（RAG）来增强 LLMs 的能力
3. 问题：**检索知识和参数化存储之间的冲突通常会误导 LLMs，挑战 RAG 系统的有效性**

### RAG 现状



