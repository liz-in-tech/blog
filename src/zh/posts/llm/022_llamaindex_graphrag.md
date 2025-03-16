---
icon: lightbulb
sidebar: false
date: 2025-02-22
prev: ./023_agent_framework
next: ./021_microsoft_graphrag
category:
  - LLM
tag:
  - GraphRAG
  - LlamaIndex
---
#  LlamaIndex + GraphRAG + Ollama + Neo4j
<!-- more -->
## 1. 官方实现改进点
- 将openai模型替换为ollama本地模型
    - llm: llama3
    - embedding model: nomic-embed-text
- 使用本地neo4j图数据库
- 优化抽取图的实体和关系的prompt以及相应的解析函数
## 2. 整体流程
- 1.加载数据，并将数据分割为文本块
- 2.从每一个文本块中抽取出实体和关系
- 3.汇总所有文本块的实体和关系，得到完整的图信息
- 4.存储图信息到Neo4j图数据库并创建索引
- 5.将图中相关的节点分组为一个个社区，并为每个社区生成摘要
- 6.检索社区摘要，从每个摘要生成答案，然后将这些答案聚合成最终响应

## 3. 实现
### 3.1. 安装需要的库
```python
!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple llama-index-graph-stores-neo4j graspologic numpy scipy==1.12.0 future 
```
### 3.2. 加载数据
#### 3.2.1. 加载csv文件，该文件有3列，分别是标题、日期和文本
```python
import pandas as pd
from llama_index.core import Document

news = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)[:10]

news.head()
```

|title|	date|	text|
|--|--|--|
|0	Chevron: Best Of Breed|	2031-04-06T01:36:32.000000000+00:00|	JHVEPhoto Like many companies in the O&G secto...|
1	FirstEnergy (NYSE:FE) Posts Earnings Results|	2030-04-29T06:55:28.000000000+00:00|	FirstEnergy (NYSE:FE – Get Rating) posted its ...|
2	Dáil almost suspended after Sinn Féin TD put p...|	2023-06-15T14:32:11.000000000+00:00|	The Dáil was almost suspended on Thursday afte...|
3	Epic’s latest tool can animate hyperrealistic ...|	2023-06-15T14:00:00.000000000+00:00|	Today, Epic is releasing a new tool designed t...|
4	EU to Ban Huawei, ZTE from Internal Commission...|	2023-06-15T13:50:00.000000000+00:00|	The European Commission is planning to ban equ...|
#### 3.2.2. 拼接title和text，得到documents
```python
documents = [Document(text=f'{row['title']}:{row['text']}') for i,row in news.iterrows()]
```
#### 3.2.3. 分割文本块
```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)
```
#### 3.2.4. 验证
```python
len(nodes)
```
> 10
```python
print(nodes[0].text)
```
> Chevron: Best Of Breed:JHVEPhoto Like many companies in the O&G sector, the stock of Chevron (NYSE:CVX) has declined about 10% over the past 90-days despite the fact that Q2 consensus earnings estimates have risen sharply (~25%) during that same time frame. Over the years, Chevron has kept a very strong balance sheet. That allowed the...

```python
print(nodes[1].text)
```
> FirstEnergy (NYSE:FE) Posts Earnings Results:FirstEnergy (NYSE:FE – Get Rating) posted its earnings results on Tuesday. The utilities provider reported $0.53 earnings per share for the quarter, topping the consensus estimate of $0.52 by $0.01, RTT News reports. FirstEnergy had a net margin of 10.85% and a return on equity of 17.17%. During the same period...
If the content contained herein violates any of your rights, including those of copyright, you are requested to immediately notify us using via the following email address operanews-external(at)opera.com
Top News

### 3.3. 抽取实体和关系
#### 3.3.1. 定义GraphRAGExtractor类
- GraphRAGExtractor类用于从文本中抽取实体 entities 和关系 relationships 的三元组subject-relation-object
- 关键组件
    - llm：抽取实体和关系采用的llm
    - extract_prompt：抽取实体和关系的prompt
    - parse_fn：将llm输出解析为结构化数据的解析函数
    - num_workers: 线程数，同时处理多个文本的数量
- 主要方法
    - __call__：处理入口
    - acall：call的异步版本
    - _aextract：核心方法
- 每个文本块的抽取过程
    - 1.将文本块和extract_prompt传给llm
    - 2.得到llm的响应结果：抽取出的实体和关系以及对它们的描述
    - 3.通过parse_fn解析llm的响应结果，得到EntityNode和Relation对象
    - 4.将所有文本块解析出的实体和关系的信息添加到KG_NODES_KEY和KG_RELATIONS_KEY的节点元数据下
```python
import asyncio
import nest_asyncio

nest_asyncio.apply()

from typing import Any, List, Callable, Optional, Union, Dict
from IPython.display import Markdown, display

from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field


class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            print(f'extract text --->:\n{text}')
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description  
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )
```

#### 3.3.2. 采用ollama本地模型，并设为全局llm
```python
import os
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

os.environ["no_proxy"] = "127.0.0.1,localhost"

llm = Ollama(model="llama3", request_timeout=660.0)

Settings.llm = llm
```
测试是否能调通
```python
response = llm.complete("What is the capital of France?")
print(response)
```
> 输出：
>
> The capital of France is Paris.

#### 3.3.3. 采用ollama本地embedding模型，并设为全局embed_model
```python
!pip install llama-index-embeddings-ollama
```

```python
from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
    request_timeout=660.0
)

# changing the global default
Settings.embed_model = ollama_embedding
```

#### 3.3.4. 定义extract_prompt
```python
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: { "entities": [], "relationships": [] }.

-An Output Example-
{
  "entities": [
    {
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics."
    },
    {
      "entity_name": "Theory of Relativity",
      "entity_type": "Scientific Theory",
      "entity_description": "A scientific theory developed by Albert Einstein, describing the laws of physics in relation to observers in different frames of reference."
    },
    {
      "entity_name": "Nobel Prize in Physics",
      "entity_type": "Award",
      "entity_description": "A prestigious international award in the field of physics, awarded annually by the Royal Swedish Academy of Sciences."
    }
  ],
  "relationships": [
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    },
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Nobel Prize in Physics",
      "relation": "won",
      "relationship_description": "Albert Einstein won the Nobel Prize in Physics in 1921."
    }
  ]
}

-Real Data-
######################
text: {text}
######################
output:"""
```
#### 3.3.5. 定义parse_fn
```python
import json
import re

def parse_fn(response_str: str) -> Any:
    print(f'parse_fn ---> response_str:\n{response_str}')
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, response_str, re.DOTALL) 
    entities = []
    relationships = []
    if not match: return entities, relationships      
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        entities = [(entity['entity_name'], entity['entity_type'], entity['entity_description']) for entity in data.get('entities', [])]
        relationships = [(relation['source_entity'], relation['target_entity'], relation['relation'], relation['relationship_description']) for relation in data.get('relationships', [])]
        print(f'parse_fn ---> entities:\n{entities}')
        print(f'parse_fn ---> relationships:\n{relationships}')
        return entities, relationships
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return entities, relationships
```
#### 3.3.6. 实例化GraphRAGExtractor为kg_extractor对象
```python    
kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)
```

### 3.4. 存储图信息到Neo4j
#### 3.4.1. 定义GraphRAGStore类
关键方法
- build_communities()
    - 将内部图表示转换为 NetworkX 图
    - 应用层次 Leiden 算法进行社区检测
    - 收集每个社区的详细信息
    - 为每个社区生成摘要
- generate_community_summary(text)
    - 使用 LLM 生成社区中关系的摘要
    - 摘要包括实体名称和关系描述的综合
- _create_nx_graph()
    - 将内部图表示转换为 NetworkX 图，以便进行社区检测
- _collect_community_info(nx_graph, clusters)
    - 根据社区收集每个节点的详细信息
    - 创建社区内每个关系的字符串表示
- _summarize_communities(community_info)
    - 使用 LLM 为每个社区生成并存储摘要
- get_community_summaries()
    - 返回社区摘要，如果尚未生成，则先构建它们

```python
import re
import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.llms import ChatMessage
from collections import defaultdict
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5
    entity_info = None

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = llm.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)
        
        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)
        
        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
```

#### 3.4.2. 实例化GraphRAGStore为graph_store对象，采用本地neo4j图数据库
```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Note: used to be `Neo4jPGStore`
graph_store = GraphRAGStore(
    username="neo4j", password="neo4j", url="bolt://localhost:7687"
)
```

### 3.5. GraphRAG 索引
#### 3.5.1. 创建索引
- nodes来自3.2.3
- kg_extractor来自3.3.6
- graph_store来自3.4.2


```python
from llama_index.core import PropertyGraphIndex

index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=graph_store,
    kg_extractors=[kg_extractor],
    show_progress=True,
)
```
每个文本块输出结果如下：
```
extract text --->:
Chevron: Best Of Breed:JHVEPhoto Like many companies in the O&G sector, the stock of Chevron (NYSE:CVX) has declined about 10% over the past 90-days despite the fact that Q2 consensus earnings estimates have risen sharply (~25%) during that same time frame. Over the years, Chevron has kept a very strong balance sheet. That allowed the...
parse_fn ---> response_str:
Here is the output for the given text:

{
  "entities": [
    {
      "entity_name": "Chevron",
      "entity_type": "Company",
      "entity_description": "A multinational energy corporation that engages in the development of crude oil and natural gas resources."
    },
    {
      "entity_name": "NYSE:CVX",
      "entity_type": "Stock Ticker Symbol",
      "entity_description": "The stock ticker symbol for Chevron Corporation's shares listed on the New York Stock Exchange."
    }
  ],
  "relationships": [
    {
      "source_entity": "Chevron",
      "target_entity": "Chevron",
      "relation": "has",
      "relationship_description": "Chevron has a strong balance sheet."
    }
  ]
}

Note that I did not identify any other entities or relationships beyond the company and its stock ticker symbol. If you would like me to look for more entities and relationships, please let me know!
parse_fn ---> entities: 
[('Chevron', 'Company', 'A multinational energy corporation that engages in the development of crude oil and natural gas resources.'), ('NYSE:CVX', 'Stock Ticker Symbol', "The stock ticker symbol for Chevron Corporation's shares listed on the New York Stock Exchange.")]
parse_fn ---> relationships: 
[('Chevron', 'Chevron', 'has', 'Chevron has a strong balance sheet.')]
```
#### 3.5.2. 验证
```python
len(index.property_graph_store.get_triplets())
```
> 148
```python
index.property_graph_store.get_triplets()[10]
```
> [EntityNode(label='Company', embedding=None, properties={'id': 'FirstEnergy', 'entity_description': 'FirstEnergy (NYSE:FE) is a utilities provider', 'triplet_source_id': '144af8a1-4078-4991-a234-4fc930bcd029'}, name='FirstEnergy'),
 Relation(label='reported_by', source_id='FirstEnergy', target_id='RTT News', properties={'relationship_description': "FirstEnergy's earnings results were reported by RTT News", 'triplet_source_id': '144af8a1-4078-4991-a234-4fc930bcd029'}),
 EntityNode(label='News Source', embedding=None, properties={'id': 'RTT News', 'entity_description': 'A news source reporting on earnings results and other financial information', 'triplet_source_id': '144af8a1-4078-4991-a234-4fc930bcd029'}, name='RTT News')]

### 3.6. 构建社区并生成社区摘要
使用社区检测算法将图中相关的节点分组，然后使用大语言模型 (LLM) 为每个社区生成摘要。

```python
index.property_graph_store.build_communities()
```

### 3.7. GraphRAG 查询
#### 3.7.1. 定义GraphRAGQueryEngine类
- GraphRAGQueryEngine 类是一个定制的查询引擎，旨在使用 GraphRAG 方法处理查询。它利用 GraphRAGStore 生成的社区摘要来回答用户查询。
- 主要组件
    - graph_store：GraphRAGStore 的实例，包含社区摘要
    - llm：用于生成和聚合答案的语言模型
- 关键方法
    - custom_query(query_str: str)
        - 查询的主入口
        - 用于检索社区摘要，从每个摘要生成答案，然后将这些答案聚合成最终响应
    - generate_answer_from_summary(community_summary, query)
        - 根据单个社区摘要为查询生成答案
        - 使用 LLM 在查询的上下文中解释社区摘要
    - aggregate_answers(community_answers)
        - 将来自不同社区的单个答案组合成一个连贯的最终响应
        - 使用 LLM 将多个视角合成为一个简洁的答案
- 查询处理流程
    - 1.从图存储中检索社区摘要
    - 2.针对每个社区摘要，为查询生成特定的答案
    - 3.将所有社区特定的答案聚合成一个最终的、连贯的响应

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from llama_index.core import PropertyGraphIndex
import re

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM
    index: PropertyGraphIndex
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        
        entities = self.get_entities(query_str, self.similarity_top_k)

        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids # 用聚类IDs进行过滤
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        enitites = set()
        pattern = (
            r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"
        )

        for node in nodes_retrieved:
            matches = re.findall(
                pattern, node.text, re.MULTILINE | re.IGNORECASE
            )

            for match in matches:
                subject = match[0]
                obj = match[2]
                enitites.add(subject)
                enitites.add(obj)

        return list(enitites)

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))
    
    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response
```


#### 3.7.2. 实例化GraphRAGQueryEngine为query_engine对象
```python
query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store, 
    llm=llm,
    index=index,
    similarity_top_k=10
)
```
#### 3.7.3. 检索信息
```python
response = query_engine.query(
    "What are the main news discussed in the document?"
)
display(Markdown(f"{response.response}"))
```
> Here is the combined response:
The main news discussed in the document is a debate on retained firefighters' pay, with Sinn Féin TDs John Brady and Pearse Doherty advocating for their rights. Specifically, John Brady interrupted Minister for Housing Darragh O'Brien's speech to emphasize the importance of retaining firefighters' pay, while Pearse Doherty called on the minister to make an improved offer in relation to pay and meet with them outside of the House.

## 4. 参考链接 
[https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/](https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/)

[https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/](https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/)