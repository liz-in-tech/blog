import{_ as e}from"./plugin-vue_export-helper-x3n3nnut.js";import{r as p,o,c as i,f as l,a as n,b as s,d as t,e as c}from"./app-xyQ2iw7y.js";const u={},r=n("h1",{id:"llamaindex-graphrag-ollama-neo4j",tabindex:"-1"},[n("a",{class:"header-anchor",href:"#llamaindex-graphrag-ollama-neo4j","aria-hidden":"true"},"#"),s(" LlamaIndex + GraphRAG + Ollama + Neo4j")],-1),d=n("ul",null,[n("li",null,"官方实现改进点"),n("li",null,"整体流程"),n("li",null,"代码实现")],-1),k=c(`<h2 id="_1-官方实现改进点" tabindex="-1"><a class="header-anchor" href="#_1-官方实现改进点" aria-hidden="true">#</a> 1. 官方实现改进点</h2><ul><li>将openai模型替换为ollama本地模型 <ul><li>llm: llama3</li><li>embedding model: nomic-embed-text</li></ul></li><li>使用本地neo4j图数据库</li><li>优化抽取图的实体和关系的prompt以及相应的解析函数</li></ul><h2 id="_2-整体流程" tabindex="-1"><a class="header-anchor" href="#_2-整体流程" aria-hidden="true">#</a> 2. 整体流程</h2><ul><li>1.加载数据，并将数据分割为文本块</li><li>2.从每一个文本块中抽取出实体和关系</li><li>3.汇总所有文本块的实体和关系，得到完整的图信息</li><li>4.存储图信息到Neo4j图数据库并创建索引</li><li>5.将图中相关的节点分组为一个个社区，并为每个社区生成摘要</li><li>6.检索社区摘要，从每个摘要生成答案，然后将这些答案聚合成最终响应</li></ul><h2 id="_3-实现" tabindex="-1"><a class="header-anchor" href="#_3-实现" aria-hidden="true">#</a> 3. 实现</h2><h3 id="_3-1-安装需要的库" tabindex="-1"><a class="header-anchor" href="#_3-1-安装需要的库" aria-hidden="true">#</a> 3.1. 安装需要的库</h3><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>!pip install <span class="token operator">-</span>i https<span class="token punctuation">:</span><span class="token operator">//</span>pypi<span class="token punctuation">.</span>tuna<span class="token punctuation">.</span>tsinghua<span class="token punctuation">.</span>edu<span class="token punctuation">.</span>cn<span class="token operator">/</span>simple llama<span class="token operator">-</span>index<span class="token operator">-</span>graph<span class="token operator">-</span>stores<span class="token operator">-</span>neo4j graspologic numpy scipy<span class="token operator">==</span><span class="token number">1.12</span><span class="token number">.0</span> future 
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_3-2-加载数据" tabindex="-1"><a class="header-anchor" href="#_3-2-加载数据" aria-hidden="true">#</a> 3.2. 加载数据</h3><h4 id="_3-2-1-加载csv文件-该文件有3列-分别是标题、日期和文本" tabindex="-1"><a class="header-anchor" href="#_3-2-1-加载csv文件-该文件有3列-分别是标题、日期和文本" aria-hidden="true">#</a> 3.2.1. 加载csv文件，该文件有3列，分别是标题、日期和文本</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core <span class="token keyword">import</span> Document

news <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span>
    <span class="token string">&quot;https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv&quot;</span>
<span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token number">10</span><span class="token punctuation">]</span>

news<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><table><thead><tr><th>title</th><th>date</th><th>text</th></tr></thead><tbody><tr><td>0 Chevron: Best Of Breed</td><td>2031-04-06T01:36:32.000000000+00:00</td><td>JHVEPhoto Like many companies in the O&amp;G secto...</td></tr><tr><td>1 FirstEnergy (NYSE:FE) Posts Earnings Results</td><td>2030-04-29T06:55:28.000000000+00:00</td><td>FirstEnergy (NYSE:FE – Get Rating) posted its ...</td></tr><tr><td>2 Dáil almost suspended after Sinn Féin TD put p...</td><td>2023-06-15T14:32:11.000000000+00:00</td><td>The Dáil was almost suspended on Thursday afte...</td></tr><tr><td>3 Epic’s latest tool can animate hyperrealistic ...</td><td>2023-06-15T14:00:00.000000000+00:00</td><td>Today, Epic is releasing a new tool designed t...</td></tr><tr><td>4 EU to Ban Huawei, ZTE from Internal Commission...</td><td>2023-06-15T13:50:00.000000000+00:00</td><td>The European Commission is planning to ban equ...</td></tr></tbody></table><h4 id="_3-2-2-拼接title和text-得到documents" tabindex="-1"><a class="header-anchor" href="#_3-2-2-拼接title和text-得到documents" aria-hidden="true">#</a> 3.2.2. 拼接title和text，得到documents</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>documents <span class="token operator">=</span> <span class="token punctuation">[</span>Document<span class="token punctuation">(</span>text<span class="token operator">=</span><span class="token string-interpolation"><span class="token string">f&#39;{row[&#39;</span></span>title<span class="token string">&#39;]}:{row[&#39;</span>text<span class="token string">&#39;]}&#39;</span><span class="token punctuation">)</span> <span class="token keyword">for</span> i<span class="token punctuation">,</span>row <span class="token keyword">in</span> news<span class="token punctuation">.</span>iterrows<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">]</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h4 id="_3-2-3-分割文本块" tabindex="-1"><a class="header-anchor" href="#_3-2-3-分割文本块" aria-hidden="true">#</a> 3.2.3. 分割文本块</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>node_parser <span class="token keyword">import</span> SentenceSplitter

splitter <span class="token operator">=</span> SentenceSplitter<span class="token punctuation">(</span>
    chunk_size<span class="token operator">=</span><span class="token number">1024</span><span class="token punctuation">,</span>
    chunk_overlap<span class="token operator">=</span><span class="token number">20</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
nodes <span class="token operator">=</span> splitter<span class="token punctuation">.</span>get_nodes_from_documents<span class="token punctuation">(</span>documents<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-2-4-验证" tabindex="-1"><a class="header-anchor" href="#_3-2-4-验证" aria-hidden="true">#</a> 3.2.4. 验证</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token builtin">len</span><span class="token punctuation">(</span>nodes<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><blockquote><p>10</p></blockquote><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">print</span><span class="token punctuation">(</span>nodes<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">.</span>text<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><blockquote><p>Chevron: Best Of Breed:JHVEPhoto Like many companies in the O&amp;G sector, the stock of Chevron (NYSE:CVX) has declined about 10% over the past 90-days despite the fact that Q2 consensus earnings estimates have risen sharply (~25%) during that same time frame. Over the years, Chevron has kept a very strong balance sheet. That allowed the...</p></blockquote><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">print</span><span class="token punctuation">(</span>nodes<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">.</span>text<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><blockquote><p>FirstEnergy (NYSE:FE) Posts Earnings Results:FirstEnergy (NYSE:FE – Get Rating) posted its earnings results on Tuesday. The utilities provider reported $0.53 earnings per share for the quarter, topping the consensus estimate of $0.52 by $0.01, RTT News reports. FirstEnergy had a net margin of 10.85% and a return on equity of 17.17%. During the same period... If the content contained herein violates any of your rights, including those of copyright, you are requested to immediately notify us using via the following email address operanews-external(at)opera.com Top News</p></blockquote><h3 id="_3-3-抽取实体和关系" tabindex="-1"><a class="header-anchor" href="#_3-3-抽取实体和关系" aria-hidden="true">#</a> 3.3. 抽取实体和关系</h3><h4 id="_3-3-1-定义graphragextractor类" tabindex="-1"><a class="header-anchor" href="#_3-3-1-定义graphragextractor类" aria-hidden="true">#</a> 3.3.1. 定义GraphRAGExtractor类</h4><ul><li>GraphRAGExtractor类用于从文本中抽取实体 entities 和关系 relationships 的三元组subject-relation-object</li><li>关键组件 <ul><li>llm：抽取实体和关系采用的llm</li><li>extract_prompt：抽取实体和关系的prompt</li><li>parse_fn：将llm输出解析为结构化数据的解析函数</li><li>num_workers: 线程数，同时处理多个文本的数量</li></ul></li><li>主要方法 <ul><li><strong>call</strong>：处理入口</li><li>acall：call的异步版本</li><li>_aextract：核心方法</li></ul></li><li>每个文本块的抽取过程 <ul><li>1.将文本块和extract_prompt传给llm</li><li>2.得到llm的响应结果：抽取出的实体和关系以及对它们的描述</li><li>3.通过parse_fn解析llm的响应结果，得到EntityNode和Relation对象</li><li>4.将所有文本块解析出的实体和关系的信息添加到KG_NODES_KEY和KG_RELATIONS_KEY的节点元数据下</li></ul></li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> asyncio
<span class="token keyword">import</span> nest_asyncio

nest_asyncio<span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">from</span> typing <span class="token keyword">import</span> Any<span class="token punctuation">,</span> List<span class="token punctuation">,</span> Callable<span class="token punctuation">,</span> Optional<span class="token punctuation">,</span> Union<span class="token punctuation">,</span> Dict
<span class="token keyword">from</span> IPython<span class="token punctuation">.</span>display <span class="token keyword">import</span> Markdown<span class="token punctuation">,</span> display

<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>async_utils <span class="token keyword">import</span> run_jobs
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>indices<span class="token punctuation">.</span>property_graph<span class="token punctuation">.</span>utils <span class="token keyword">import</span> <span class="token punctuation">(</span>
    default_parse_triplets_fn<span class="token punctuation">,</span>
<span class="token punctuation">)</span>
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>graph_stores<span class="token punctuation">.</span>types <span class="token keyword">import</span> <span class="token punctuation">(</span>
    EntityNode<span class="token punctuation">,</span>
    KG_NODES_KEY<span class="token punctuation">,</span>
    KG_RELATIONS_KEY<span class="token punctuation">,</span>
    Relation<span class="token punctuation">,</span>
<span class="token punctuation">)</span>
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>llms<span class="token punctuation">.</span>llm <span class="token keyword">import</span> LLM
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>prompts <span class="token keyword">import</span> PromptTemplate
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>prompts<span class="token punctuation">.</span>default_prompts <span class="token keyword">import</span> <span class="token punctuation">(</span>
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT<span class="token punctuation">,</span>
<span class="token punctuation">)</span>
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>schema <span class="token keyword">import</span> TransformComponent<span class="token punctuation">,</span> BaseNode
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>bridge<span class="token punctuation">.</span>pydantic <span class="token keyword">import</span> BaseModel<span class="token punctuation">,</span> Field


<span class="token keyword">class</span> <span class="token class-name">GraphRAGExtractor</span><span class="token punctuation">(</span>TransformComponent<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token triple-quoted-string string">&quot;&quot;&quot;Extract triples from a graph.

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
    &quot;&quot;&quot;</span>

    llm<span class="token punctuation">:</span> LLM
    extract_prompt<span class="token punctuation">:</span> PromptTemplate
    parse_fn<span class="token punctuation">:</span> Callable
    num_workers<span class="token punctuation">:</span> <span class="token builtin">int</span>
    max_paths_per_chunk<span class="token punctuation">:</span> <span class="token builtin">int</span>

    <span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>
        self<span class="token punctuation">,</span>
        llm<span class="token punctuation">:</span> Optional<span class="token punctuation">[</span>LLM<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span>
        extract_prompt<span class="token punctuation">:</span> Optional<span class="token punctuation">[</span>Union<span class="token punctuation">[</span><span class="token builtin">str</span><span class="token punctuation">,</span> PromptTemplate<span class="token punctuation">]</span><span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span>
        parse_fn<span class="token punctuation">:</span> Callable <span class="token operator">=</span> default_parse_triplets_fn<span class="token punctuation">,</span>
        max_paths_per_chunk<span class="token punctuation">:</span> <span class="token builtin">int</span> <span class="token operator">=</span> <span class="token number">10</span><span class="token punctuation">,</span>
        num_workers<span class="token punctuation">:</span> <span class="token builtin">int</span> <span class="token operator">=</span> <span class="token number">4</span><span class="token punctuation">,</span>
    <span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token boolean">None</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Init params.&quot;&quot;&quot;</span>
        <span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core <span class="token keyword">import</span> Settings

        <span class="token keyword">if</span> <span class="token builtin">isinstance</span><span class="token punctuation">(</span>extract_prompt<span class="token punctuation">,</span> <span class="token builtin">str</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
            extract_prompt <span class="token operator">=</span> PromptTemplate<span class="token punctuation">(</span>extract_prompt<span class="token punctuation">)</span>

        <span class="token builtin">super</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>__init__<span class="token punctuation">(</span>
            llm<span class="token operator">=</span>llm <span class="token keyword">or</span> Settings<span class="token punctuation">.</span>llm<span class="token punctuation">,</span>
            extract_prompt<span class="token operator">=</span>extract_prompt <span class="token keyword">or</span> DEFAULT_KG_TRIPLET_EXTRACT_PROMPT<span class="token punctuation">,</span>
            parse_fn<span class="token operator">=</span>parse_fn<span class="token punctuation">,</span>
            num_workers<span class="token operator">=</span>num_workers<span class="token punctuation">,</span>
            max_paths_per_chunk<span class="token operator">=</span>max_paths_per_chunk<span class="token punctuation">,</span>
        <span class="token punctuation">)</span>

    <span class="token decorator annotation punctuation">@classmethod</span>
    <span class="token keyword">def</span> <span class="token function">class_name</span><span class="token punctuation">(</span>cls<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">str</span><span class="token punctuation">:</span>
        <span class="token keyword">return</span> <span class="token string">&quot;GraphExtractor&quot;</span>

    <span class="token keyword">def</span> <span class="token function">__call__</span><span class="token punctuation">(</span>
        self<span class="token punctuation">,</span> nodes<span class="token punctuation">:</span> List<span class="token punctuation">[</span>BaseNode<span class="token punctuation">]</span><span class="token punctuation">,</span> show_progress<span class="token punctuation">:</span> <span class="token builtin">bool</span> <span class="token operator">=</span> <span class="token boolean">False</span><span class="token punctuation">,</span> <span class="token operator">**</span>kwargs<span class="token punctuation">:</span> Any
    <span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span>BaseNode<span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Extract triples from nodes.&quot;&quot;&quot;</span>
        <span class="token keyword">return</span> asyncio<span class="token punctuation">.</span>run<span class="token punctuation">(</span>
            self<span class="token punctuation">.</span>acall<span class="token punctuation">(</span>nodes<span class="token punctuation">,</span> show_progress<span class="token operator">=</span>show_progress<span class="token punctuation">,</span> <span class="token operator">**</span>kwargs<span class="token punctuation">)</span>
        <span class="token punctuation">)</span>

    <span class="token keyword">async</span> <span class="token keyword">def</span> <span class="token function">_aextract</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> node<span class="token punctuation">:</span> BaseNode<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> BaseNode<span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Extract triples from a node.&quot;&quot;&quot;</span>
        <span class="token keyword">assert</span> <span class="token builtin">hasattr</span><span class="token punctuation">(</span>node<span class="token punctuation">,</span> <span class="token string">&quot;text&quot;</span><span class="token punctuation">)</span>

        text <span class="token operator">=</span> node<span class="token punctuation">.</span>get_content<span class="token punctuation">(</span>metadata_mode<span class="token operator">=</span><span class="token string">&quot;llm&quot;</span><span class="token punctuation">)</span>
        <span class="token keyword">try</span><span class="token punctuation">:</span>
            llm_response <span class="token operator">=</span> <span class="token keyword">await</span> self<span class="token punctuation">.</span>llm<span class="token punctuation">.</span>apredict<span class="token punctuation">(</span>
                self<span class="token punctuation">.</span>extract_prompt<span class="token punctuation">,</span>
                text<span class="token operator">=</span>text<span class="token punctuation">,</span>
                max_knowledge_triplets<span class="token operator">=</span>self<span class="token punctuation">.</span>max_paths_per_chunk<span class="token punctuation">,</span>
            <span class="token punctuation">)</span>
            <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&#39;extract text ---&gt;:\\n</span><span class="token interpolation"><span class="token punctuation">{</span>text<span class="token punctuation">}</span></span><span class="token string">&#39;</span></span><span class="token punctuation">)</span>
            entities<span class="token punctuation">,</span> entities_relationship <span class="token operator">=</span> self<span class="token punctuation">.</span>parse_fn<span class="token punctuation">(</span>llm_response<span class="token punctuation">)</span>
        <span class="token keyword">except</span> ValueError<span class="token punctuation">:</span>
            entities <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
            entities_relationship <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

        existing_nodes <span class="token operator">=</span> node<span class="token punctuation">.</span>metadata<span class="token punctuation">.</span>pop<span class="token punctuation">(</span>KG_NODES_KEY<span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
        existing_relations <span class="token operator">=</span> node<span class="token punctuation">.</span>metadata<span class="token punctuation">.</span>pop<span class="token punctuation">(</span>KG_RELATIONS_KEY<span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
        entity_metadata <span class="token operator">=</span> node<span class="token punctuation">.</span>metadata<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">for</span> entity<span class="token punctuation">,</span> entity_type<span class="token punctuation">,</span> description <span class="token keyword">in</span> entities<span class="token punctuation">:</span>
            entity_metadata<span class="token punctuation">[</span><span class="token string">&quot;entity_description&quot;</span><span class="token punctuation">]</span> <span class="token operator">=</span> description  
            entity_node <span class="token operator">=</span> EntityNode<span class="token punctuation">(</span>
                name<span class="token operator">=</span>entity<span class="token punctuation">,</span> label<span class="token operator">=</span>entity_type<span class="token punctuation">,</span> properties<span class="token operator">=</span>entity_metadata
            <span class="token punctuation">)</span>
            existing_nodes<span class="token punctuation">.</span>append<span class="token punctuation">(</span>entity_node<span class="token punctuation">)</span>

        relation_metadata <span class="token operator">=</span> node<span class="token punctuation">.</span>metadata<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">for</span> triple <span class="token keyword">in</span> entities_relationship<span class="token punctuation">:</span>
            subj<span class="token punctuation">,</span> obj<span class="token punctuation">,</span> rel<span class="token punctuation">,</span> description <span class="token operator">=</span> triple
            relation_metadata<span class="token punctuation">[</span><span class="token string">&quot;relationship_description&quot;</span><span class="token punctuation">]</span> <span class="token operator">=</span> description
            rel_node <span class="token operator">=</span> Relation<span class="token punctuation">(</span>
                label<span class="token operator">=</span>rel<span class="token punctuation">,</span>
                source_id<span class="token operator">=</span>subj<span class="token punctuation">,</span>
                target_id<span class="token operator">=</span>obj<span class="token punctuation">,</span>
                properties<span class="token operator">=</span>relation_metadata<span class="token punctuation">,</span>
            <span class="token punctuation">)</span>
            existing_relations<span class="token punctuation">.</span>append<span class="token punctuation">(</span>rel_node<span class="token punctuation">)</span>

        node<span class="token punctuation">.</span>metadata<span class="token punctuation">[</span>KG_NODES_KEY<span class="token punctuation">]</span> <span class="token operator">=</span> existing_nodes
        node<span class="token punctuation">.</span>metadata<span class="token punctuation">[</span>KG_RELATIONS_KEY<span class="token punctuation">]</span> <span class="token operator">=</span> existing_relations
        <span class="token keyword">return</span> node

    <span class="token keyword">async</span> <span class="token keyword">def</span> <span class="token function">acall</span><span class="token punctuation">(</span>
        self<span class="token punctuation">,</span> nodes<span class="token punctuation">:</span> List<span class="token punctuation">[</span>BaseNode<span class="token punctuation">]</span><span class="token punctuation">,</span> show_progress<span class="token punctuation">:</span> <span class="token builtin">bool</span> <span class="token operator">=</span> <span class="token boolean">False</span><span class="token punctuation">,</span> <span class="token operator">**</span>kwargs<span class="token punctuation">:</span> Any
    <span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> List<span class="token punctuation">[</span>BaseNode<span class="token punctuation">]</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Extract triples from nodes async.&quot;&quot;&quot;</span>
        jobs <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
        <span class="token keyword">for</span> node <span class="token keyword">in</span> nodes<span class="token punctuation">:</span>
            jobs<span class="token punctuation">.</span>append<span class="token punctuation">(</span>self<span class="token punctuation">.</span>_aextract<span class="token punctuation">(</span>node<span class="token punctuation">)</span><span class="token punctuation">)</span>

        <span class="token keyword">return</span> <span class="token keyword">await</span> run_jobs<span class="token punctuation">(</span>
            jobs<span class="token punctuation">,</span>
            workers<span class="token operator">=</span>self<span class="token punctuation">.</span>num_workers<span class="token punctuation">,</span>
            show_progress<span class="token operator">=</span>show_progress<span class="token punctuation">,</span>
            desc<span class="token operator">=</span><span class="token string">&quot;Extracting paths from text&quot;</span><span class="token punctuation">,</span>
        <span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-3-2-采用ollama本地模型-并设为全局llm" tabindex="-1"><a class="header-anchor" href="#_3-3-2-采用ollama本地模型-并设为全局llm" aria-hidden="true">#</a> 3.3.2. 采用ollama本地模型，并设为全局llm</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> os
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>llms<span class="token punctuation">.</span>ollama <span class="token keyword">import</span> Ollama
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core <span class="token keyword">import</span> Settings

os<span class="token punctuation">.</span>environ<span class="token punctuation">[</span><span class="token string">&quot;no_proxy&quot;</span><span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token string">&quot;127.0.0.1,localhost&quot;</span>

llm <span class="token operator">=</span> Ollama<span class="token punctuation">(</span>model<span class="token operator">=</span><span class="token string">&quot;llama3&quot;</span><span class="token punctuation">,</span> request_timeout<span class="token operator">=</span><span class="token number">660.0</span><span class="token punctuation">)</span>

Settings<span class="token punctuation">.</span>llm <span class="token operator">=</span> llm
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>测试是否能调通</p><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>response <span class="token operator">=</span> llm<span class="token punctuation">.</span>complete<span class="token punctuation">(</span><span class="token string">&quot;What is the capital of France?&quot;</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>response<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><blockquote><p>输出：</p><p>The capital of France is Paris.</p></blockquote><h4 id="_3-3-3-采用ollama本地embedding模型-并设为全局embed-model" tabindex="-1"><a class="header-anchor" href="#_3-3-3-采用ollama本地embedding模型-并设为全局embed-model" aria-hidden="true">#</a> 3.3.3. 采用ollama本地embedding模型，并设为全局embed_model</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>!pip install llama<span class="token operator">-</span>index<span class="token operator">-</span>embeddings<span class="token operator">-</span>ollama
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>embeddings<span class="token punctuation">.</span>ollama <span class="token keyword">import</span> OllamaEmbedding

ollama_embedding <span class="token operator">=</span> OllamaEmbedding<span class="token punctuation">(</span>
    model_name<span class="token operator">=</span><span class="token string">&quot;nomic-embed-text&quot;</span><span class="token punctuation">,</span>
    base_url<span class="token operator">=</span><span class="token string">&quot;http://localhost:11434&quot;</span><span class="token punctuation">,</span>
    ollama_additional_kwargs<span class="token operator">=</span><span class="token punctuation">{</span><span class="token string">&quot;mirostat&quot;</span><span class="token punctuation">:</span> <span class="token number">0</span><span class="token punctuation">}</span><span class="token punctuation">,</span>
    request_timeout<span class="token operator">=</span><span class="token number">660.0</span>
<span class="token punctuation">)</span>

<span class="token comment"># changing the global default</span>
Settings<span class="token punctuation">.</span>embed_model <span class="token operator">=</span> ollama_embedding
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-3-4-定义extract-prompt" tabindex="-1"><a class="header-anchor" href="#_3-3-4-定义extract-prompt" aria-hidden="true">#</a> 3.3.4. 定义extract_prompt</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>KG_TRIPLET_EXTRACT_TMPL <span class="token operator">=</span> <span class="token triple-quoted-string string">&quot;&quot;&quot;
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity&#39;s attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: &#39;entities&#39; (list of entity objects) and &#39;relationships&#39; (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: { &quot;entities&quot;: [], &quot;relationships&quot;: [] }.

-An Output Example-
{
  &quot;entities&quot;: [
    {
      &quot;entity_name&quot;: &quot;Albert Einstein&quot;,
      &quot;entity_type&quot;: &quot;Person&quot;,
      &quot;entity_description&quot;: &quot;Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics.&quot;
    },
    {
      &quot;entity_name&quot;: &quot;Theory of Relativity&quot;,
      &quot;entity_type&quot;: &quot;Scientific Theory&quot;,
      &quot;entity_description&quot;: &quot;A scientific theory developed by Albert Einstein, describing the laws of physics in relation to observers in different frames of reference.&quot;
    },
    {
      &quot;entity_name&quot;: &quot;Nobel Prize in Physics&quot;,
      &quot;entity_type&quot;: &quot;Award&quot;,
      &quot;entity_description&quot;: &quot;A prestigious international award in the field of physics, awarded annually by the Royal Swedish Academy of Sciences.&quot;
    }
  ],
  &quot;relationships&quot;: [
    {
      &quot;source_entity&quot;: &quot;Albert Einstein&quot;,
      &quot;target_entity&quot;: &quot;Theory of Relativity&quot;,
      &quot;relation&quot;: &quot;developed&quot;,
      &quot;relationship_description&quot;: &quot;Albert Einstein is the developer of the theory of relativity.&quot;
    },
    {
      &quot;source_entity&quot;: &quot;Albert Einstein&quot;,
      &quot;target_entity&quot;: &quot;Nobel Prize in Physics&quot;,
      &quot;relation&quot;: &quot;won&quot;,
      &quot;relationship_description&quot;: &quot;Albert Einstein won the Nobel Prize in Physics in 1921.&quot;
    }
  ]
}

-Real Data-
######################
text: {text}
######################
output:&quot;&quot;&quot;</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-3-5-定义parse-fn" tabindex="-1"><a class="header-anchor" href="#_3-3-5-定义parse-fn" aria-hidden="true">#</a> 3.3.5. 定义parse_fn</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> json
<span class="token keyword">import</span> re

<span class="token keyword">def</span> <span class="token function">parse_fn</span><span class="token punctuation">(</span>response_str<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> Any<span class="token punctuation">:</span>
    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&#39;parse_fn ---&gt; response_str:\\n</span><span class="token interpolation"><span class="token punctuation">{</span>response_str<span class="token punctuation">}</span></span><span class="token string">&#39;</span></span><span class="token punctuation">)</span>
    json_pattern <span class="token operator">=</span> <span class="token string">r&#39;\\{.*\\}&#39;</span>
    <span class="token keyword">match</span> <span class="token operator">=</span> re<span class="token punctuation">.</span>search<span class="token punctuation">(</span>json_pattern<span class="token punctuation">,</span> response_str<span class="token punctuation">,</span> re<span class="token punctuation">.</span>DOTALL<span class="token punctuation">)</span> 
    entities <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    relationships <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    <span class="token keyword">if</span> <span class="token keyword">not</span> <span class="token keyword">match</span><span class="token punctuation">:</span> <span class="token keyword">return</span> entities<span class="token punctuation">,</span> relationships      
    json_str <span class="token operator">=</span> <span class="token keyword">match</span><span class="token punctuation">.</span>group<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span>
    <span class="token keyword">try</span><span class="token punctuation">:</span>
        data <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>json_str<span class="token punctuation">)</span>
        entities <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">(</span>entity<span class="token punctuation">[</span><span class="token string">&#39;entity_name&#39;</span><span class="token punctuation">]</span><span class="token punctuation">,</span> entity<span class="token punctuation">[</span><span class="token string">&#39;entity_type&#39;</span><span class="token punctuation">]</span><span class="token punctuation">,</span> entity<span class="token punctuation">[</span><span class="token string">&#39;entity_description&#39;</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token keyword">for</span> entity <span class="token keyword">in</span> data<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">&#39;entities&#39;</span><span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">]</span>
        relationships <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">(</span>relation<span class="token punctuation">[</span><span class="token string">&#39;source_entity&#39;</span><span class="token punctuation">]</span><span class="token punctuation">,</span> relation<span class="token punctuation">[</span><span class="token string">&#39;target_entity&#39;</span><span class="token punctuation">]</span><span class="token punctuation">,</span> relation<span class="token punctuation">[</span><span class="token string">&#39;relation&#39;</span><span class="token punctuation">]</span><span class="token punctuation">,</span> relation<span class="token punctuation">[</span><span class="token string">&#39;relationship_description&#39;</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token keyword">for</span> relation <span class="token keyword">in</span> data<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">&#39;relationships&#39;</span><span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">]</span>
        <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&#39;parse_fn ---&gt; entities:\\n</span><span class="token interpolation"><span class="token punctuation">{</span>entities<span class="token punctuation">}</span></span><span class="token string">&#39;</span></span><span class="token punctuation">)</span>
        <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&#39;parse_fn ---&gt; relationships:\\n</span><span class="token interpolation"><span class="token punctuation">{</span>relationships<span class="token punctuation">}</span></span><span class="token string">&#39;</span></span><span class="token punctuation">)</span>
        <span class="token keyword">return</span> entities<span class="token punctuation">,</span> relationships
    <span class="token keyword">except</span> json<span class="token punctuation">.</span>JSONDecodeError <span class="token keyword">as</span> e<span class="token punctuation">:</span>
        <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">&quot;Error parsing JSON:&quot;</span><span class="token punctuation">,</span> e<span class="token punctuation">)</span>
        <span class="token keyword">return</span> entities<span class="token punctuation">,</span> relationships
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-3-6-实例化graphragextractor为kg-extractor对象" tabindex="-1"><a class="header-anchor" href="#_3-3-6-实例化graphragextractor为kg-extractor对象" aria-hidden="true">#</a> 3.3.6. 实例化GraphRAGExtractor为kg_extractor对象</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>kg_extractor <span class="token operator">=</span> GraphRAGExtractor<span class="token punctuation">(</span>
    llm<span class="token operator">=</span>llm<span class="token punctuation">,</span>
    extract_prompt<span class="token operator">=</span>KG_TRIPLET_EXTRACT_TMPL<span class="token punctuation">,</span>
    max_paths_per_chunk<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">,</span>
    parse_fn<span class="token operator">=</span>parse_fn<span class="token punctuation">,</span>
<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_3-4-存储图信息到neo4j" tabindex="-1"><a class="header-anchor" href="#_3-4-存储图信息到neo4j" aria-hidden="true">#</a> 3.4. 存储图信息到Neo4j</h3><h4 id="_3-4-1-定义graphragstore类" tabindex="-1"><a class="header-anchor" href="#_3-4-1-定义graphragstore类" aria-hidden="true">#</a> 3.4.1. 定义GraphRAGStore类</h4><p>关键方法</p><ul><li>build_communities() <ul><li>将内部图表示转换为 NetworkX 图</li><li>应用层次 Leiden 算法进行社区检测</li><li>收集每个社区的详细信息</li><li>为每个社区生成摘要</li></ul></li><li>generate_community_summary(text) <ul><li>使用 LLM 生成社区中关系的摘要</li><li>摘要包括实体名称和关系描述的综合</li></ul></li><li>_create_nx_graph() <ul><li>将内部图表示转换为 NetworkX 图，以便进行社区检测</li></ul></li><li>_collect_community_info(nx_graph, clusters) <ul><li>根据社区收集每个节点的详细信息</li><li>创建社区内每个关系的字符串表示</li></ul></li><li>_summarize_communities(community_info) <ul><li>使用 LLM 为每个社区生成并存储摘要</li></ul></li><li>get_community_summaries() <ul><li>返回社区摘要，如果尚未生成，则先构建它们</li></ul></li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">import</span> re
<span class="token keyword">import</span> networkx <span class="token keyword">as</span> nx
<span class="token keyword">from</span> graspologic<span class="token punctuation">.</span>partition <span class="token keyword">import</span> hierarchical_leiden
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>llms <span class="token keyword">import</span> ChatMessage
<span class="token keyword">from</span> collections <span class="token keyword">import</span> defaultdict
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>graph_stores<span class="token punctuation">.</span>neo4j <span class="token keyword">import</span> Neo4jPropertyGraphStore


<span class="token keyword">class</span> <span class="token class-name">GraphRAGStore</span><span class="token punctuation">(</span>Neo4jPropertyGraphStore<span class="token punctuation">)</span><span class="token punctuation">:</span>
    community_summary <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span>
    max_cluster_size <span class="token operator">=</span> <span class="token number">5</span>
    entity_info <span class="token operator">=</span> <span class="token boolean">None</span>

    <span class="token keyword">def</span> <span class="token function">generate_community_summary</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> text<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Generate summary for a given text using an LLM.&quot;&quot;&quot;</span>
        messages <span class="token operator">=</span> <span class="token punctuation">[</span>
            ChatMessage<span class="token punctuation">(</span>
                role<span class="token operator">=</span><span class="token string">&quot;system&quot;</span><span class="token punctuation">,</span>
                content<span class="token operator">=</span><span class="token punctuation">(</span>
                    <span class="token string">&quot;You are provided with a set of relationships from a knowledge graph, each represented as &quot;</span>
                    <span class="token string">&quot;entity1-&gt;entity2-&gt;relation-&gt;relationship_description. Your task is to create a summary of these &quot;</span>
                    <span class="token string">&quot;relationships. The summary should include the names of the entities involved and a concise synthesis &quot;</span>
                    <span class="token string">&quot;of the relationship descriptions. The goal is to capture the most critical and relevant details that &quot;</span>
                    <span class="token string">&quot;highlight the nature and significance of each relationship. Ensure that the summary is coherent and &quot;</span>
                    <span class="token string">&quot;integrates the information in a way that emphasizes the key aspects of the relationships.&quot;</span>
                <span class="token punctuation">)</span><span class="token punctuation">,</span>
            <span class="token punctuation">)</span><span class="token punctuation">,</span>
            ChatMessage<span class="token punctuation">(</span>role<span class="token operator">=</span><span class="token string">&quot;user&quot;</span><span class="token punctuation">,</span> content<span class="token operator">=</span>text<span class="token punctuation">)</span><span class="token punctuation">,</span>
        <span class="token punctuation">]</span>
        response <span class="token operator">=</span> llm<span class="token punctuation">.</span>chat<span class="token punctuation">(</span>messages<span class="token punctuation">)</span>
        clean_response <span class="token operator">=</span> re<span class="token punctuation">.</span>sub<span class="token punctuation">(</span><span class="token string">r&quot;^assistant:\\s*&quot;</span><span class="token punctuation">,</span> <span class="token string">&quot;&quot;</span><span class="token punctuation">,</span> <span class="token builtin">str</span><span class="token punctuation">(</span>response<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">return</span> clean_response

    <span class="token keyword">def</span> <span class="token function">build_communities</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Builds communities from the graph and summarizes them.&quot;&quot;&quot;</span>
        nx_graph <span class="token operator">=</span> self<span class="token punctuation">.</span>_create_nx_graph<span class="token punctuation">(</span><span class="token punctuation">)</span>
        community_hierarchical_clusters <span class="token operator">=</span> hierarchical_leiden<span class="token punctuation">(</span>
            nx_graph<span class="token punctuation">,</span> max_cluster_size<span class="token operator">=</span>self<span class="token punctuation">.</span>max_cluster_size
        <span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>entity_info<span class="token punctuation">,</span> community_info <span class="token operator">=</span> self<span class="token punctuation">.</span>_collect_community_info<span class="token punctuation">(</span>
            nx_graph<span class="token punctuation">,</span> community_hierarchical_clusters
        <span class="token punctuation">)</span>
        self<span class="token punctuation">.</span>_summarize_communities<span class="token punctuation">(</span>community_info<span class="token punctuation">)</span>

    <span class="token keyword">def</span> <span class="token function">_create_nx_graph</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Converts internal graph representation to NetworkX graph.&quot;&quot;&quot;</span>
        nx_graph <span class="token operator">=</span> nx<span class="token punctuation">.</span>Graph<span class="token punctuation">(</span><span class="token punctuation">)</span>
        triplets <span class="token operator">=</span> self<span class="token punctuation">.</span>get_triplets<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">for</span> entity1<span class="token punctuation">,</span> relation<span class="token punctuation">,</span> entity2 <span class="token keyword">in</span> triplets<span class="token punctuation">:</span>
            nx_graph<span class="token punctuation">.</span>add_node<span class="token punctuation">(</span>entity1<span class="token punctuation">.</span>name<span class="token punctuation">)</span>
            nx_graph<span class="token punctuation">.</span>add_node<span class="token punctuation">(</span>entity2<span class="token punctuation">.</span>name<span class="token punctuation">)</span>
            nx_graph<span class="token punctuation">.</span>add_edge<span class="token punctuation">(</span>
                relation<span class="token punctuation">.</span>source_id<span class="token punctuation">,</span>
                relation<span class="token punctuation">.</span>target_id<span class="token punctuation">,</span>
                relationship<span class="token operator">=</span>relation<span class="token punctuation">.</span>label<span class="token punctuation">,</span>
                description<span class="token operator">=</span>relation<span class="token punctuation">.</span>properties<span class="token punctuation">[</span><span class="token string">&quot;relationship_description&quot;</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
            <span class="token punctuation">)</span>
        <span class="token keyword">return</span> nx_graph

    <span class="token keyword">def</span> <span class="token function">_collect_community_info</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> nx_graph<span class="token punctuation">,</span> clusters<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        &quot;&quot;&quot;</span>
        entity_info <span class="token operator">=</span> defaultdict<span class="token punctuation">(</span><span class="token builtin">set</span><span class="token punctuation">)</span>
        community_info <span class="token operator">=</span> defaultdict<span class="token punctuation">(</span><span class="token builtin">list</span><span class="token punctuation">)</span>
        
        <span class="token keyword">for</span> item <span class="token keyword">in</span> clusters<span class="token punctuation">:</span>
            node <span class="token operator">=</span> item<span class="token punctuation">.</span>node
            cluster_id <span class="token operator">=</span> item<span class="token punctuation">.</span>cluster

            <span class="token comment"># Update entity_info</span>
            entity_info<span class="token punctuation">[</span>node<span class="token punctuation">]</span><span class="token punctuation">.</span>add<span class="token punctuation">(</span>cluster_id<span class="token punctuation">)</span>

            <span class="token keyword">for</span> neighbor <span class="token keyword">in</span> nx_graph<span class="token punctuation">.</span>neighbors<span class="token punctuation">(</span>node<span class="token punctuation">)</span><span class="token punctuation">:</span>
                edge_data <span class="token operator">=</span> nx_graph<span class="token punctuation">.</span>get_edge_data<span class="token punctuation">(</span>node<span class="token punctuation">,</span> neighbor<span class="token punctuation">)</span>
                <span class="token keyword">if</span> edge_data<span class="token punctuation">:</span>
                    detail <span class="token operator">=</span> <span class="token string-interpolation"><span class="token string">f&quot;</span><span class="token interpolation"><span class="token punctuation">{</span>node<span class="token punctuation">}</span></span><span class="token string"> -&gt; </span><span class="token interpolation"><span class="token punctuation">{</span>neighbor<span class="token punctuation">}</span></span><span class="token string"> -&gt; </span><span class="token interpolation"><span class="token punctuation">{</span>edge_data<span class="token punctuation">[</span><span class="token string">&#39;relationship&#39;</span><span class="token punctuation">]</span><span class="token punctuation">}</span></span><span class="token string"> -&gt; </span><span class="token interpolation"><span class="token punctuation">{</span>edge_data<span class="token punctuation">[</span><span class="token string">&#39;description&#39;</span><span class="token punctuation">]</span><span class="token punctuation">}</span></span><span class="token string">&quot;</span></span>
                    community_info<span class="token punctuation">[</span>cluster_id<span class="token punctuation">]</span><span class="token punctuation">.</span>append<span class="token punctuation">(</span>detail<span class="token punctuation">)</span>
        
        <span class="token comment"># Convert sets to lists for easier serialization if needed</span>
        entity_info <span class="token operator">=</span> <span class="token punctuation">{</span>k<span class="token punctuation">:</span> <span class="token builtin">list</span><span class="token punctuation">(</span>v<span class="token punctuation">)</span> <span class="token keyword">for</span> k<span class="token punctuation">,</span> v <span class="token keyword">in</span> entity_info<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">}</span>

        <span class="token keyword">return</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>entity_info<span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token builtin">dict</span><span class="token punctuation">(</span>community_info<span class="token punctuation">)</span>

    <span class="token keyword">def</span> <span class="token function">_summarize_communities</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> community_info<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Generate and store summaries for each community.&quot;&quot;&quot;</span>
        <span class="token keyword">for</span> community_id<span class="token punctuation">,</span> details <span class="token keyword">in</span> community_info<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
            details_text <span class="token operator">=</span> <span class="token punctuation">(</span>
                <span class="token string">&quot;\\n&quot;</span><span class="token punctuation">.</span>join<span class="token punctuation">(</span>details<span class="token punctuation">)</span> <span class="token operator">+</span> <span class="token string">&quot;.&quot;</span>
            <span class="token punctuation">)</span>  <span class="token comment"># Ensure it ends with a period</span>
            self<span class="token punctuation">.</span>community_summary<span class="token punctuation">[</span>
                community_id
            <span class="token punctuation">]</span> <span class="token operator">=</span> self<span class="token punctuation">.</span>generate_community_summary<span class="token punctuation">(</span>details_text<span class="token punctuation">)</span>

    <span class="token keyword">def</span> <span class="token function">get_community_summaries</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Returns the community summaries, building them if not already done.&quot;&quot;&quot;</span>
        <span class="token keyword">if</span> <span class="token keyword">not</span> self<span class="token punctuation">.</span>community_summary<span class="token punctuation">:</span>
            self<span class="token punctuation">.</span>build_communities<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">return</span> self<span class="token punctuation">.</span>community_summary
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-4-2-实例化graphragstore为graph-store对象-采用本地neo4j图数据库" tabindex="-1"><a class="header-anchor" href="#_3-4-2-实例化graphragstore为graph-store对象-采用本地neo4j图数据库" aria-hidden="true">#</a> 3.4.2. 实例化GraphRAGStore为graph_store对象，采用本地neo4j图数据库</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>graph_stores<span class="token punctuation">.</span>neo4j <span class="token keyword">import</span> Neo4jPropertyGraphStore

<span class="token comment"># Note: used to be \`Neo4jPGStore\`</span>
graph_store <span class="token operator">=</span> GraphRAGStore<span class="token punctuation">(</span>
    username<span class="token operator">=</span><span class="token string">&quot;neo4j&quot;</span><span class="token punctuation">,</span> password<span class="token operator">=</span><span class="token string">&quot;neo4j&quot;</span><span class="token punctuation">,</span> url<span class="token operator">=</span><span class="token string">&quot;bolt://localhost:7687&quot;</span>
<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_3-5-graphrag-索引" tabindex="-1"><a class="header-anchor" href="#_3-5-graphrag-索引" aria-hidden="true">#</a> 3.5. GraphRAG 索引</h3><h4 id="_3-5-1-创建索引" tabindex="-1"><a class="header-anchor" href="#_3-5-1-创建索引" aria-hidden="true">#</a> 3.5.1. 创建索引</h4><ul><li>nodes来自3.2.3</li><li>kg_extractor来自3.3.6</li><li>graph_store来自3.4.2</li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core <span class="token keyword">import</span> PropertyGraphIndex

index <span class="token operator">=</span> PropertyGraphIndex<span class="token punctuation">(</span>
    nodes<span class="token operator">=</span>nodes<span class="token punctuation">,</span>
    property_graph_store<span class="token operator">=</span>graph_store<span class="token punctuation">,</span>
    kg_extractors<span class="token operator">=</span><span class="token punctuation">[</span>kg_extractor<span class="token punctuation">]</span><span class="token punctuation">,</span>
    show_progress<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>每个文本块输出结果如下：</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>extract text ---&gt;:
Chevron: Best Of Breed:JHVEPhoto Like many companies in the O&amp;G sector, the stock of Chevron (NYSE:CVX) has declined about 10% over the past 90-days despite the fact that Q2 consensus earnings estimates have risen sharply (~25%) during that same time frame. Over the years, Chevron has kept a very strong balance sheet. That allowed the...
parse_fn ---&gt; response_str:
Here is the output for the given text:

{
  &quot;entities&quot;: [
    {
      &quot;entity_name&quot;: &quot;Chevron&quot;,
      &quot;entity_type&quot;: &quot;Company&quot;,
      &quot;entity_description&quot;: &quot;A multinational energy corporation that engages in the development of crude oil and natural gas resources.&quot;
    },
    {
      &quot;entity_name&quot;: &quot;NYSE:CVX&quot;,
      &quot;entity_type&quot;: &quot;Stock Ticker Symbol&quot;,
      &quot;entity_description&quot;: &quot;The stock ticker symbol for Chevron Corporation&#39;s shares listed on the New York Stock Exchange.&quot;
    }
  ],
  &quot;relationships&quot;: [
    {
      &quot;source_entity&quot;: &quot;Chevron&quot;,
      &quot;target_entity&quot;: &quot;Chevron&quot;,
      &quot;relation&quot;: &quot;has&quot;,
      &quot;relationship_description&quot;: &quot;Chevron has a strong balance sheet.&quot;
    }
  ]
}

Note that I did not identify any other entities or relationships beyond the company and its stock ticker symbol. If you would like me to look for more entities and relationships, please let me know!
parse_fn ---&gt; entities: 
[(&#39;Chevron&#39;, &#39;Company&#39;, &#39;A multinational energy corporation that engages in the development of crude oil and natural gas resources.&#39;), (&#39;NYSE:CVX&#39;, &#39;Stock Ticker Symbol&#39;, &quot;The stock ticker symbol for Chevron Corporation&#39;s shares listed on the New York Stock Exchange.&quot;)]
parse_fn ---&gt; relationships: 
[(&#39;Chevron&#39;, &#39;Chevron&#39;, &#39;has&#39;, &#39;Chevron has a strong balance sheet.&#39;)]
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-5-2-验证" tabindex="-1"><a class="header-anchor" href="#_3-5-2-验证" aria-hidden="true">#</a> 3.5.2. 验证</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token builtin">len</span><span class="token punctuation">(</span>index<span class="token punctuation">.</span>property_graph_store<span class="token punctuation">.</span>get_triplets<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><blockquote><p>148</p></blockquote><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>index<span class="token punctuation">.</span>property_graph_store<span class="token punctuation">.</span>get_triplets<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">10</span><span class="token punctuation">]</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><blockquote><p>[EntityNode(label=&#39;Company&#39;, embedding=None, properties={&#39;id&#39;: &#39;FirstEnergy&#39;, &#39;entity_description&#39;: &#39;FirstEnergy (NYSE:FE) is a utilities provider&#39;, &#39;triplet_source_id&#39;: &#39;144af8a1-4078-4991-a234-4fc930bcd029&#39;}, name=&#39;FirstEnergy&#39;), Relation(label=&#39;reported_by&#39;, source_id=&#39;FirstEnergy&#39;, target_id=&#39;RTT News&#39;, properties={&#39;relationship_description&#39;: &quot;FirstEnergy&#39;s earnings results were reported by RTT News&quot;, &#39;triplet_source_id&#39;: &#39;144af8a1-4078-4991-a234-4fc930bcd029&#39;}), EntityNode(label=&#39;News Source&#39;, embedding=None, properties={&#39;id&#39;: &#39;RTT News&#39;, &#39;entity_description&#39;: &#39;A news source reporting on earnings results and other financial information&#39;, &#39;triplet_source_id&#39;: &#39;144af8a1-4078-4991-a234-4fc930bcd029&#39;}, name=&#39;RTT News&#39;)]</p></blockquote><h3 id="_3-6-构建社区并生成社区摘要" tabindex="-1"><a class="header-anchor" href="#_3-6-构建社区并生成社区摘要" aria-hidden="true">#</a> 3.6. 构建社区并生成社区摘要</h3><p>使用社区检测算法将图中相关的节点分组，然后使用大语言模型 (LLM) 为每个社区生成摘要。</p><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>index<span class="token punctuation">.</span>property_graph_store<span class="token punctuation">.</span>build_communities<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_3-7-graphrag-查询" tabindex="-1"><a class="header-anchor" href="#_3-7-graphrag-查询" aria-hidden="true">#</a> 3.7. GraphRAG 查询</h3><h4 id="_3-7-1-定义graphragqueryengine类" tabindex="-1"><a class="header-anchor" href="#_3-7-1-定义graphragqueryengine类" aria-hidden="true">#</a> 3.7.1. 定义GraphRAGQueryEngine类</h4><ul><li>GraphRAGQueryEngine 类是一个定制的查询引擎，旨在使用 GraphRAG 方法处理查询。它利用 GraphRAGStore 生成的社区摘要来回答用户查询。</li><li>主要组件 <ul><li>graph_store：GraphRAGStore 的实例，包含社区摘要</li><li>llm：用于生成和聚合答案的语言模型</li></ul></li><li>关键方法 <ul><li>custom_query(query_str: str) <ul><li>查询的主入口</li><li>用于检索社区摘要，从每个摘要生成答案，然后将这些答案聚合成最终响应</li></ul></li><li>generate_answer_from_summary(community_summary, query) <ul><li>根据单个社区摘要为查询生成答案</li><li>使用 LLM 在查询的上下文中解释社区摘要</li></ul></li><li>aggregate_answers(community_answers) <ul><li>将来自不同社区的单个答案组合成一个连贯的最终响应</li><li>使用 LLM 将多个视角合成为一个简洁的答案</li></ul></li></ul></li><li>查询处理流程 <ul><li>1.从图存储中检索社区摘要</li><li>2.针对每个社区摘要，为查询生成特定的答案</li><li>3.将所有社区特定的答案聚合成一个最终的、连贯的响应</li></ul></li></ul><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code><span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>query_engine <span class="token keyword">import</span> CustomQueryEngine
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core<span class="token punctuation">.</span>llms <span class="token keyword">import</span> LLM
<span class="token keyword">from</span> llama_index<span class="token punctuation">.</span>core <span class="token keyword">import</span> PropertyGraphIndex
<span class="token keyword">import</span> re

<span class="token keyword">class</span> <span class="token class-name">GraphRAGQueryEngine</span><span class="token punctuation">(</span>CustomQueryEngine<span class="token punctuation">)</span><span class="token punctuation">:</span>
    graph_store<span class="token punctuation">:</span> GraphRAGStore
    llm<span class="token punctuation">:</span> LLM
    index<span class="token punctuation">:</span> PropertyGraphIndex
    similarity_top_k<span class="token punctuation">:</span> <span class="token builtin">int</span> <span class="token operator">=</span> <span class="token number">20</span>

    <span class="token keyword">def</span> <span class="token function">custom_query</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> query_str<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">&gt;</span> <span class="token builtin">str</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Process all community summaries to generate answers to a specific query.&quot;&quot;&quot;</span>
        
        entities <span class="token operator">=</span> self<span class="token punctuation">.</span>get_entities<span class="token punctuation">(</span>query_str<span class="token punctuation">,</span> self<span class="token punctuation">.</span>similarity_top_k<span class="token punctuation">)</span>

        community_ids <span class="token operator">=</span> self<span class="token punctuation">.</span>retrieve_entity_communities<span class="token punctuation">(</span>
            self<span class="token punctuation">.</span>graph_store<span class="token punctuation">.</span>entity_info<span class="token punctuation">,</span> entities
        <span class="token punctuation">)</span>
        
        community_summaries <span class="token operator">=</span> self<span class="token punctuation">.</span>graph_store<span class="token punctuation">.</span>get_community_summaries<span class="token punctuation">(</span><span class="token punctuation">)</span>
        community_answers <span class="token operator">=</span> <span class="token punctuation">[</span>
            self<span class="token punctuation">.</span>generate_answer_from_summary<span class="token punctuation">(</span>community_summary<span class="token punctuation">,</span> query_str<span class="token punctuation">)</span>
            <span class="token keyword">for</span> <span class="token builtin">id</span><span class="token punctuation">,</span> community_summary <span class="token keyword">in</span> community_summaries<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span>
            <span class="token keyword">if</span> <span class="token builtin">id</span> <span class="token keyword">in</span> community_ids <span class="token comment"># 用聚类IDs进行过滤</span>
        <span class="token punctuation">]</span>

        final_answer <span class="token operator">=</span> self<span class="token punctuation">.</span>aggregate_answers<span class="token punctuation">(</span>community_answers<span class="token punctuation">)</span>
        <span class="token keyword">return</span> final_answer

    <span class="token keyword">def</span> <span class="token function">get_entities</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> query_str<span class="token punctuation">,</span> similarity_top_k<span class="token punctuation">)</span><span class="token punctuation">:</span>
        nodes_retrieved <span class="token operator">=</span> self<span class="token punctuation">.</span>index<span class="token punctuation">.</span>as_retriever<span class="token punctuation">(</span>
            similarity_top_k<span class="token operator">=</span>similarity_top_k
        <span class="token punctuation">)</span><span class="token punctuation">.</span>retrieve<span class="token punctuation">(</span>query_str<span class="token punctuation">)</span>

        enitites <span class="token operator">=</span> <span class="token builtin">set</span><span class="token punctuation">(</span><span class="token punctuation">)</span>
        pattern <span class="token operator">=</span> <span class="token punctuation">(</span>
            <span class="token string">r&quot;^(\\w+(?:\\s+\\w+)*)\\s*-&gt;\\s*([a-zA-Z\\s]+?)\\s*-&gt;\\s*(\\w+(?:\\s+\\w+)*)$&quot;</span>
        <span class="token punctuation">)</span>

        <span class="token keyword">for</span> node <span class="token keyword">in</span> nodes_retrieved<span class="token punctuation">:</span>
            matches <span class="token operator">=</span> re<span class="token punctuation">.</span>findall<span class="token punctuation">(</span>
                pattern<span class="token punctuation">,</span> node<span class="token punctuation">.</span>text<span class="token punctuation">,</span> re<span class="token punctuation">.</span>MULTILINE <span class="token operator">|</span> re<span class="token punctuation">.</span>IGNORECASE
            <span class="token punctuation">)</span>

            <span class="token keyword">for</span> <span class="token keyword">match</span> <span class="token keyword">in</span> matches<span class="token punctuation">:</span>
                subject <span class="token operator">=</span> <span class="token keyword">match</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span>
                obj <span class="token operator">=</span> <span class="token keyword">match</span><span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">]</span>
                enitites<span class="token punctuation">.</span>add<span class="token punctuation">(</span>subject<span class="token punctuation">)</span>
                enitites<span class="token punctuation">.</span>add<span class="token punctuation">(</span>obj<span class="token punctuation">)</span>

        <span class="token keyword">return</span> <span class="token builtin">list</span><span class="token punctuation">(</span>enitites<span class="token punctuation">)</span>

    <span class="token keyword">def</span> <span class="token function">retrieve_entity_communities</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> entity_info<span class="token punctuation">,</span> entities<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        &quot;&quot;&quot;</span>
        community_ids <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

        <span class="token keyword">for</span> entity <span class="token keyword">in</span> entities<span class="token punctuation">:</span>
            <span class="token keyword">if</span> entity <span class="token keyword">in</span> entity_info<span class="token punctuation">:</span>
                community_ids<span class="token punctuation">.</span>extend<span class="token punctuation">(</span>entity_info<span class="token punctuation">[</span>entity<span class="token punctuation">]</span><span class="token punctuation">)</span>

        <span class="token keyword">return</span> <span class="token builtin">list</span><span class="token punctuation">(</span><span class="token builtin">set</span><span class="token punctuation">(</span>community_ids<span class="token punctuation">)</span><span class="token punctuation">)</span>
    
    <span class="token keyword">def</span> <span class="token function">generate_answer_from_summary</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> community_summary<span class="token punctuation">,</span> query<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Generate an answer from a community summary based on a given query using LLM.&quot;&quot;&quot;</span>
        prompt <span class="token operator">=</span> <span class="token punctuation">(</span>
            <span class="token string-interpolation"><span class="token string">f&quot;Given the community summary: </span><span class="token interpolation"><span class="token punctuation">{</span>community_summary<span class="token punctuation">}</span></span><span class="token string">, &quot;</span></span>
            <span class="token string-interpolation"><span class="token string">f&quot;how would you answer the following query? Query: </span><span class="token interpolation"><span class="token punctuation">{</span>query<span class="token punctuation">}</span></span><span class="token string">&quot;</span></span>
        <span class="token punctuation">)</span>
        messages <span class="token operator">=</span> <span class="token punctuation">[</span>
            ChatMessage<span class="token punctuation">(</span>role<span class="token operator">=</span><span class="token string">&quot;system&quot;</span><span class="token punctuation">,</span> content<span class="token operator">=</span>prompt<span class="token punctuation">)</span><span class="token punctuation">,</span>
            ChatMessage<span class="token punctuation">(</span>
                role<span class="token operator">=</span><span class="token string">&quot;user&quot;</span><span class="token punctuation">,</span>
                content<span class="token operator">=</span><span class="token string">&quot;I need an answer based on the above information.&quot;</span><span class="token punctuation">,</span>
            <span class="token punctuation">)</span><span class="token punctuation">,</span>
        <span class="token punctuation">]</span>
        response <span class="token operator">=</span> self<span class="token punctuation">.</span>llm<span class="token punctuation">.</span>chat<span class="token punctuation">(</span>messages<span class="token punctuation">)</span>
        cleaned_response <span class="token operator">=</span> re<span class="token punctuation">.</span>sub<span class="token punctuation">(</span><span class="token string">r&quot;^assistant:\\s*&quot;</span><span class="token punctuation">,</span> <span class="token string">&quot;&quot;</span><span class="token punctuation">,</span> <span class="token builtin">str</span><span class="token punctuation">(</span>response<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">return</span> cleaned_response

    <span class="token keyword">def</span> <span class="token function">aggregate_answers</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> community_answers<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token triple-quoted-string string">&quot;&quot;&quot;Aggregate individual community answers into a final, coherent response.&quot;&quot;&quot;</span>
        <span class="token comment"># intermediate_text = &quot; &quot;.join(community_answers)</span>
        prompt <span class="token operator">=</span> <span class="token string">&quot;Combine the following intermediate answers into a final, concise response.&quot;</span>
        messages <span class="token operator">=</span> <span class="token punctuation">[</span>
            ChatMessage<span class="token punctuation">(</span>role<span class="token operator">=</span><span class="token string">&quot;system&quot;</span><span class="token punctuation">,</span> content<span class="token operator">=</span>prompt<span class="token punctuation">)</span><span class="token punctuation">,</span>
            ChatMessage<span class="token punctuation">(</span>
                role<span class="token operator">=</span><span class="token string">&quot;user&quot;</span><span class="token punctuation">,</span>
                content<span class="token operator">=</span><span class="token string-interpolation"><span class="token string">f&quot;Intermediate answers: </span><span class="token interpolation"><span class="token punctuation">{</span>community_answers<span class="token punctuation">}</span></span><span class="token string">&quot;</span></span><span class="token punctuation">,</span>
            <span class="token punctuation">)</span><span class="token punctuation">,</span>
        <span class="token punctuation">]</span>
        final_response <span class="token operator">=</span> self<span class="token punctuation">.</span>llm<span class="token punctuation">.</span>chat<span class="token punctuation">(</span>messages<span class="token punctuation">)</span>
        cleaned_final_response <span class="token operator">=</span> re<span class="token punctuation">.</span>sub<span class="token punctuation">(</span>
            <span class="token string">r&quot;^assistant:\\s*&quot;</span><span class="token punctuation">,</span> <span class="token string">&quot;&quot;</span><span class="token punctuation">,</span> <span class="token builtin">str</span><span class="token punctuation">(</span>final_response<span class="token punctuation">)</span>
        <span class="token punctuation">)</span><span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span>
        <span class="token keyword">return</span> cleaned_final_response
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-7-2-实例化graphragqueryengine为query-engine对象" tabindex="-1"><a class="header-anchor" href="#_3-7-2-实例化graphragqueryengine为query-engine对象" aria-hidden="true">#</a> 3.7.2. 实例化GraphRAGQueryEngine为query_engine对象</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>query_engine <span class="token operator">=</span> GraphRAGQueryEngine<span class="token punctuation">(</span>
    graph_store<span class="token operator">=</span>index<span class="token punctuation">.</span>property_graph_store<span class="token punctuation">,</span> 
    llm<span class="token operator">=</span>llm<span class="token punctuation">,</span>
    index<span class="token operator">=</span>index<span class="token punctuation">,</span>
    similarity_top_k<span class="token operator">=</span><span class="token number">10</span>
<span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h4 id="_3-7-3-检索信息" tabindex="-1"><a class="header-anchor" href="#_3-7-3-检索信息" aria-hidden="true">#</a> 3.7.3. 检索信息</h4><div class="language-python line-numbers-mode" data-ext="py"><pre class="language-python"><code>response <span class="token operator">=</span> query_engine<span class="token punctuation">.</span>query<span class="token punctuation">(</span>
    <span class="token string">&quot;What are the main news discussed in the document?&quot;</span>
<span class="token punctuation">)</span>
display<span class="token punctuation">(</span>Markdown<span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f&quot;</span><span class="token interpolation"><span class="token punctuation">{</span>response<span class="token punctuation">.</span>response<span class="token punctuation">}</span></span><span class="token string">&quot;</span></span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><blockquote><p>Here is the combined response: The main news discussed in the document is a debate on retained firefighters&#39; pay, with Sinn Féin TDs John Brady and Pearse Doherty advocating for their rights. Specifically, John Brady interrupted Minister for Housing Darragh O&#39;Brien&#39;s speech to emphasize the importance of retaining firefighters&#39; pay, while Pearse Doherty called on the minister to make an improved offer in relation to pay and meet with them outside of the House.</p></blockquote><h2 id="_4-参考链接" tabindex="-1"><a class="header-anchor" href="#_4-参考链接" aria-hidden="true">#</a> 4. 参考链接</h2>`,71),m={href:"https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/",target:"_blank",rel:"noopener noreferrer"},v={href:"https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/",target:"_blank",rel:"noopener noreferrer"};function b(h,y){const a=p("ExternalLinkIcon");return o(),i("div",null,[r,d,l(" more "),k,n("p",null,[n("a",m,[s("https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/"),t(a)])]),n("p",null,[n("a",v,[s("https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/"),t(a)])])])}const q=e(u,[["render",b],["__file","022_llamaindex_graphrag.html.vue"]]);export{q as default};
