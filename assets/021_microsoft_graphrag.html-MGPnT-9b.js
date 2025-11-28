import{_ as t,a,b as s,c as r}from"./021_graphrag_drift_search-N3Np96tC.js";import{_ as d}from"./plugin-vue_export-helper-x3n3nnut.js";import{r as u,o,c,f as v,a as i,b as e,d as l,e as m}from"./app-hKujL8Ix.js";const _={},p=i("h1",{id:"微软graphrag源码解读",tabindex:"-1"},[i("a",{class:"header-anchor",href:"#微软graphrag源码解读","aria-hidden":"true"},"#"),e(" 微软GraphRAG源码解读")],-1),b=i("ul",null,[i("li",null,"源码入口"),i("li",null,"init 源码逻辑"),i("li",null,"index 源码逻辑"),i("li",null,"query 源码逻辑")],-1),q=i("h2",{id:"_1-官方文档",tabindex:"-1"},[i("a",{class:"header-anchor",href:"#_1-官方文档","aria-hidden":"true"},"#"),e(" 1. 官方文档")],-1),h={href:"https://github.com/microsoft/graphrag",target:"_blank",rel:"noopener noreferrer"},g={href:"https://microsoft.github.io/graphrag/get_started/",target:"_blank",rel:"noopener noreferrer"},x=m(`<p>Paper: https://arxiv.org/pdf/2404.16130</p><h2 id="_2-源码入口" tabindex="-1"><a class="header-anchor" href="#_2-源码入口" aria-hidden="true">#</a> 2. 源码入口</h2><blockquote><p>graphrag.cli.main.py</p></blockquote><p>5个函数对应5个命令</p><ul><li>init =&gt; def _initialize_cli() <ul><li>graphrag init --root ./ragtest</li></ul></li><li>index =&gt; def _index_cli() <ul><li>graphrag index --root ./ragtest</li></ul></li><li>query =&gt; def _query_cli() <ul><li>global search <ul><li>graphrag query --root ./ragtest --method global --query &quot;What are the top themes in this story?&quot;</li></ul></li><li>local search <ul><li>graphrag query --root ./ragtest --method local --query &quot;Who is Scrooge and what are his main relationships?&quot;</li></ul></li></ul></li><li>update =&gt; def _update_cli() <ul><li>graphrag update</li></ul></li><li>prompt-tune =&gt; def _prompt_tune_cli() <ul><li>graphrag prompt-tune</li></ul></li></ul><p>原理：使用Python的Typer库来创建命令行接口(CLI)，以index为例</p><ul><li>@app.command(&quot;index&quot;) 是装饰器语法，用于将下面的_index_cli函数注册为CLI命令，创建一个名为&quot;index&quot;的子命令</li><li>当用户在命令行运行&quot;graphrag index&quot;时，就会调用被装饰的def _index_cli()函数</li><li>Typer会自动将函数参数转换为命令行选项，并处理参数解析和验证</li></ul><h3 id="_2-1-init" tabindex="-1"><a class="header-anchor" href="#_2-1-init" aria-hidden="true">#</a> 2.1. init</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>@app.command(&quot;init&quot;)
def _initialize_cli(
    root, 
    force,
):
    &quot;&quot;&quot;Generate a default configuration file.&quot;&quot;&quot;
    from graphrag.cli.initialize import initialize_project_at

    initialize_project_at(path=root, force=force)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_2-2-index" tabindex="-1"><a class="header-anchor" href="#_2-2-index" aria-hidden="true">#</a> 2.2. index</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>@app.command(&quot;index&quot;)
def _index_cli(
    config,
    root,  
    method,
    verbose,
    memprofile,
    logger,
    dry_run,
    cache,
    skip_validation,
    output,
):
    &quot;&quot;&quot;Build a knowledge graph index.&quot;&quot;&quot;
    from graphrag.cli.index import index_cli

    index_cli(
        root_dir=root,
        verbose=verbose,
        memprofile=memprofile,
        cache=cache,
        logger=LoggerType(logger),
        config_filepath=config,
        dry_run=dry_run,
        skip_validation=skip_validation,
        output_dir=output,
        method=method,
    )
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_2-3-qurey" tabindex="-1"><a class="header-anchor" href="#_2-3-qurey" aria-hidden="true">#</a> 2.3. qurey</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>@app.command(&quot;query&quot;)
def _query_cli(
    method,
    query,
    config,
    data,
    root,  
    community_level,
    dynamic_community_selection,
    response_type,
    streaming,
):
    &quot;&quot;&quot;Query a knowledge graph index.&quot;&quot;&quot;
    from graphrag.cli.query import (
        run_basic_search,
        run_drift_search,
        run_global_search,
        run_local_search,
    )

    match method:
        case SearchMethod.LOCAL:
            run_local_search(
                config_filepath=config,
                data_dir=data,
                root_dir=root,
                community_level=community_level,
                response_type=response_type,
                streaming=streaming,
                query=query,
            )
        case SearchMethod.GLOBAL:
            run_global_search(
                config_filepath=config,
                data_dir=data,
                root_dir=root,
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                streaming=streaming,
                query=query,
            )
        case SearchMethod.DRIFT:
            run_drift_search(
                config_filepath=config,
                data_dir=data,
                root_dir=root,
                community_level=community_level,
                streaming=streaming,
                response_type=response_type,
                query=query,
            )
        case SearchMethod.BASIC:
            run_basic_search(
                config_filepath=config,
                data_dir=data,
                root_dir=root,
                streaming=streaming,
                query=query,
            )
        case _:
            raise ValueError(INVALID_METHOD_ERROR)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="_3-init-源码逻辑" tabindex="-1"><a class="header-anchor" href="#_3-init-源码逻辑" aria-hidden="true">#</a> 3. init 源码逻辑</h2><ul><li>在指定的根目录下生成初始化文件 <ul><li>生成配置文件 settings.yaml</li><li>生成环境变量文件 .env</li><li>在根目录的prompts路径下生成多个xxx_prompt.txt文件</li></ul></li></ul><h2 id="_4-index-源码逻辑" tabindex="-1"><a class="header-anchor" href="#_4-index-源码逻辑" aria-hidden="true">#</a> 4. index 源码逻辑</h2><figure><img src="`+t+`" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_4-1-加载配置文件" tabindex="-1"><a class="header-anchor" href="#_4-1-加载配置文件" aria-hidden="true">#</a> 4.1. 加载配置文件</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.cli.index.py -&gt; def index_cli()

config = load_config(root_dir, config_filepath, cli_overrides)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-2-构建索引" tabindex="-1"><a class="header-anchor" href="#_4-2-构建索引" aria-hidden="true">#</a> 4.2. 构建索引</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.api.index.py -&gt; def build_index()

async def build_index(
    config: GraphRagConfig,
    method: IndexingMethod = IndexingMethod.Standard,
    is_update_run: bool = False,
    memory_profile: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    progress_logger: ProgressLogger | None = None,
) -&gt; list[PipelineRunResult]:

    # 创建pipeline
    pipeline = PipelineFactory.create_pipeline(config, method)

    # 执行pipeline
    async for output in run_pipeline(
        pipeline,
        config,
        cache=pipeline_cache,
        callbacks=callbacks,
        logger=progress_logger,
        is_update_run=is_update_run,
    ):
        outputs.append(output)
    return outputs
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-3-创建pipeline" tabindex="-1"><a class="header-anchor" href="#_4-3-创建pipeline" aria-hidden="true">#</a> 4.3. 创建pipeline</h3><p>参数 method 决定 pipeline 类型</p><ul><li>标准Pipeline: IndexingMethod.Standard =&gt; full LLM <ul><li>&quot;create_base_text_units&quot;,</li><li>&quot;create_final_documents&quot;,</li><li>&quot;extract_graph&quot;,</li><li>&quot;finalize_graph&quot;,</li><li>*([&quot;extract_covariates&quot;] if config.extract_claims.enabled else []),</li><li>&quot;create_communities&quot;,</li><li>&quot;create_final_text_units&quot;,</li><li>&quot;create_community_reports&quot;,</li><li>&quot;generate_text_embeddings&quot;,</li></ul></li><li>快速Pipeline: IndexingMethod.Fast =&gt; NLP + LLM <ul><li>&quot;create_base_text_units&quot;,</li><li>&quot;create_final_documents&quot;,</li><li>&quot;extract_graph_nlp&quot;,</li><li>&quot;prune_graph&quot;,</li><li>&quot;finalize_graph&quot;,</li><li>&quot;create_communities&quot;,</li><li>&quot;create_final_text_units&quot;,</li><li>&quot;create_community_reports_text&quot;,</li><li>&quot;generate_text_embeddings&quot;,</li></ul></li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.workflows.factory.py -&gt; def create_pipeline()

class PipelineFactory:
    &quot;&quot;&quot;A factory class for workflow pipelines.&quot;&quot;&quot;

    workflows: ClassVar[dict[str, WorkflowFunction]] = {}

    @classmethod
    def create_pipeline(
        cls, config: GraphRagConfig, method: IndexingMethod = IndexingMethod.Standard
    ) -&gt; Pipeline:
        &quot;&quot;&quot;Create a pipeline generator.&quot;&quot;&quot;
        workflows = _get_workflows_list(config, method)
        for name in workflows:
            yield name, cls.workflows[name]


def _get_workflows_list(
    config: GraphRagConfig, method: IndexingMethod = IndexingMethod.Standard
) -&gt; list[str]:
    &quot;&quot;&quot;Return a list of workflows for the indexing pipeline.&quot;&quot;&quot;
    if config.workflows:
        return config.workflows
    match method:
        case IndexingMethod.Standard:
            return [
                &quot;create_base_text_units&quot;,
                &quot;create_final_documents&quot;,
                &quot;extract_graph&quot;,
                &quot;finalize_graph&quot;,
                *([&quot;extract_covariates&quot;] if config.extract_claims.enabled else []),
                &quot;create_communities&quot;,
                &quot;create_final_text_units&quot;,
                &quot;create_community_reports&quot;,
                &quot;generate_text_embeddings&quot;,
            ]
        case IndexingMethod.Fast:
            return [
                &quot;create_base_text_units&quot;,
                &quot;create_final_documents&quot;,
                &quot;extract_graph_nlp&quot;,
                &quot;prune_graph&quot;,
                &quot;finalize_graph&quot;,
                &quot;create_communities&quot;,
                &quot;create_final_text_units&quot;,
                &quot;create_community_reports_text&quot;,
                &quot;generate_text_embeddings&quot;,
            ]
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-4-加载input数据集" tabindex="-1"><a class="header-anchor" href="#_4-4-加载input数据集" aria-hidden="true">#</a> 4.4. 加载Input数据集</h3><ul><li>加载根目录input文件夹下的文件</li><li>目前支持txt和csv文件类型</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.run.run_pipeline.py -&gt; def run_pipeline()

dataset = await create_input(config.input, logger, root_dir)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-5-生成documents-parquet文件" tabindex="-1"><a class="header-anchor" href="#_4-5-生成documents-parquet文件" aria-hidden="true">#</a> 4.5. 生成documents.parquet文件</h3><ul><li>行：input文件夹下每一个文件是一行</li><li>列： <ul><li>&#39;text&#39; : 文档里的文本内容</li><li>&#39;id&#39; : 文档ID</li><li>&#39;title&#39; : 文档名</li><li>&#39;creation_date&#39; : 创建时间</li></ul></li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.run.run_pipeline.py -&gt; def _run_pipeline()

await write_table_to_storage(dataset, &quot;documents&quot;, context.storage)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-6-执行pipeline-create-base-text-units-文档分块" tabindex="-1"><a class="header-anchor" href="#_4-6-执行pipeline-create-base-text-units-文档分块" aria-hidden="true">#</a> 4.6. 执行pipeline : create_base_text_units（文档分块）</h3><ul><li>加载documents.parquet文件</li><li>生成text_unit.parquet文件 <ul><li>行：将documents.parquet每行进行分块，每一行为一个分块</li><li>列： <ul><li>&#39;id&#39; : 分块ID</li><li>&#39;text&#39; : 分块文本</li><li>&#39;document_ids&#39; : 所属文件ID</li><li>&#39;n_tokens&#39; : token数</li></ul></li></ul></li></ul><h3 id="_4-7-执行pipeline-create-final-documents-聚合分块id列表到documents-parquet文件" tabindex="-1"><a class="header-anchor" href="#_4-7-执行pipeline-create-final-documents-聚合分块id列表到documents-parquet文件" aria-hidden="true">#</a> 4.7. 执行pipeline : create_final_documents（聚合分块ID列表到documents.parquet文件）</h3><ul><li>加载documents.parquet文件和text_unit.parquet文件</li><li>create_final_documents <ul><li>以文档ID为关联inner连接合并documents.parquet和text_unit.parquet文件</li><li>按文档ID进行分组，聚合块ID作为text_unit_ids列</li></ul></li><li>更新documents.parquet文件 <ul><li>行：input文件夹下每一个文件是一行</li><li>列： <ul><li>&#39;id&#39; : 文档ID (多位无序的唯一ID)</li><li>&#39;human_readable_id&#39; : 可读的序号ID (按顺序编号的ID)</li><li>&#39;text&#39; : 文档里的文本内容</li><li>&#39;text_unit_id&#39; : 分块ID列表</li><li>&#39;creation_date&#39; : 创建时间</li></ul></li></ul></li></ul><h3 id="_4-8-执行pipeline-extract-graph" tabindex="-1"><a class="header-anchor" href="#_4-8-执行pipeline-extract-graph" aria-hidden="true">#</a> 4.8. 执行pipeline : extract_graph</h3><ul><li>加载text_unit.parquet文件</li><li>extract_graph (详见以下代码) <ul><li>load_llm</li><li>run_extract_graph 抽取出实体和关系 (详见以下代码)</li><li>run_summarize_descriptions 生成摘要描述 (详见以下代码)</li><li>return (entities, relationships) <ul><li>entities <ul><li>&quot;title&quot;, 实体名称</li><li>&quot;type&quot;, 实体类型</li><li>&quot;description&quot;, 实体描述</li><li>&quot;text_unit_ids&quot;, 分块ID列表</li><li>&quot;frequency&quot;, 实体出现的次数</li></ul></li><li>relationships <ul><li>&quot;source&quot;, 关系的起始实体ID</li><li>&quot;target&quot;, 关系的结束实体ID</li><li>&quot;description&quot;, 关系的描述</li><li>&quot;weight&quot;, 关系权重（连接强度）</li><li>&quot;text_unit_ids&quot;, 分块ID列表</li></ul></li></ul></li></ul></li><li>生成entities.parquet和relationships.parquet文件</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>graphrag.index.workflows.extract_graph.py -&gt; def extract_graph()

async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    extraction_strategy: dict[str, Any] | None = None,
    extraction_num_threads: int = 4,
    extraction_async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types: list[str] | None = None,
    summarization_strategy: dict[str, Any] | None = None,
    summarization_num_threads: int = 4,
) -&gt; tuple[pd.DataFrame, pd.DataFrame]:
    &quot;&quot;&quot;All the steps to create the base entity graph.&quot;&quot;&quot;
    # this returns a graph for each text unit, to be merged later
    extracted_entities, extracted_relationships = await extractor(
        text_units=text_units,
        callbacks=callbacks,
        cache=cache,
        text_column=&quot;text&quot;,
        id_column=&quot;id&quot;,
        strategy=extraction_strategy,
        async_mode=extraction_async_mode,
        entity_types=entity_types,
        num_threads=extraction_num_threads,
    )

    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=callbacks,
        cache=cache,
        strategy=summarization_strategy,
        num_threads=summarization_num_threads,
    )

    relationships = extracted_relationships.drop(columns=[&quot;description&quot;]).merge(
        relationship_summaries, on=[&quot;source&quot;, &quot;target&quot;], how=&quot;left&quot;
    )

    extracted_entities.drop(columns=[&quot;description&quot;], inplace=True)
    entities = extracted_entities.merge(entity_summaries, on=&quot;title&quot;, how=&quot;left&quot;)

    return (entities, relationships)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.operations.extract_graph.graph_intelligence_strategy.py -&gt; def run_extract_graph()

async def run_extract_graph(
    llm: ChatLLM,
    docs: list[Document],
    entity_types: EntityTypes,
    callbacks: WorkflowCallbacks | None,
    args: StrategyConfig,
) -&gt; EntityExtractionResult:
    &quot;&quot;&quot;Run the entity extraction chain.&quot;&quot;&quot;
    tuple_delimiter = args.get(&quot;tuple_delimiter&quot;, None)
    record_delimiter = args.get(&quot;record_delimiter&quot;, None)
    completion_delimiter = args.get(&quot;completion_delimiter&quot;, None)
    extraction_prompt = args.get(&quot;extraction_prompt&quot;, None)
    encoding_model = args.get(&quot;encoding_name&quot;, None)
    max_gleanings = args.get(&quot;max_gleanings&quot;, defs.EXTRACT_GRAPH_MAX_GLEANINGS)

    extractor = GraphExtractor(
        llm_invoker=llm,
        prompt=extraction_prompt,
        encoding_model=encoding_model,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: (
            callbacks.error(&quot;Entity Extraction Error&quot;, e, s, d) if callbacks else None
        ),
    )
    text_list = [doc.text.strip() for doc in docs]

    results = await extractor(
        list(text_list),
        {
            &quot;entity_types&quot;: entity_types,
            &quot;tuple_delimiter&quot;: tuple_delimiter,
            &quot;record_delimiter&quot;: record_delimiter,
            &quot;completion_delimiter&quot;: completion_delimiter,
        },
    )

    graph = results.output
    # Map the &quot;source_id&quot; back to the &quot;id&quot; field
    for _, node in graph.nodes(data=True):  # type: ignore
        if node is not None:
            node[&quot;source_id&quot;] = &quot;,&quot;.join(
                docs[int(id)].id for id in node[&quot;source_id&quot;].split(&quot;,&quot;)
            )

    for _, _, edge in graph.edges(data=True):  # type: ignore
        if edge is not None:
            edge[&quot;source_id&quot;] = &quot;,&quot;.join(
                docs[int(id)].id for id in edge[&quot;source_id&quot;].split(&quot;,&quot;)
            )

    entities = [
        ({&quot;title&quot;: item[0], **(item[1] or {})})
        for item in graph.nodes(data=True)
        if item is not None
    ]

    relationships = nx.to_pandas_edgelist(graph)

    return EntityExtractionResult(entities, relationships, graph)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.operations.summarize_descriptions.graph_intelligence_strategy.py -&gt; def run_summarize_descriptions()

async def run_summarize_descriptions(
    llm: ChatLLM,
    id: str | tuple[str, str],
    descriptions: list[str],
    callbacks: WorkflowCallbacks,
    args: StrategyConfig,
) -&gt; SummarizedDescriptionResult:
    &quot;&quot;&quot;Run the entity extraction chain.&quot;&quot;&quot;
    # Extraction Arguments
    summarize_prompt = args.get(&quot;summarize_prompt&quot;, None)
    entity_name_key = args.get(&quot;entity_name_key&quot;, &quot;entity_name&quot;)
    input_descriptions_key = args.get(&quot;input_descriptions_key&quot;, &quot;description_list&quot;)
    max_tokens = args.get(&quot;max_tokens&quot;, None)

    extractor = SummarizeExtractor(
        llm_invoker=llm,
        summarization_prompt=summarize_prompt,
        entity_name_key=entity_name_key,
        input_descriptions_key=input_descriptions_key,
        on_error=lambda e, stack, details: (
            callbacks.error(&quot;Entity Extraction Error&quot;, e, stack, details)
            if callbacks
            else None
        ),
        max_summary_length=args.get(&quot;max_summary_length&quot;, None),
        max_input_tokens=max_tokens,
    )

    result = await extractor(id=id, descriptions=descriptions)
    return SummarizedDescriptionResult(id=result.id, description=result.description)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-9-执行pipeline-finalize-graph" tabindex="-1"><a class="header-anchor" href="#_4-9-执行pipeline-finalize-graph" aria-hidden="true">#</a> 4.9. 执行pipeline : finalize_graph</h3><ul><li>加载entities.parquet和relationships.parquet文件</li><li>finalize_graph <ul><li>finalize_entities <ul><li>&quot;id&quot;, 实体ID</li><li>&quot;human_readable_id&quot;, 可读的有序实体ID</li><li>&quot;title&quot;, 实体名称</li><li>&quot;type&quot;, 实体类型</li><li>&quot;description&quot;, 实体描述</li><li>&quot;text_unit_ids&quot;, 分块ID列表</li><li>&quot;frequency&quot;, 实体出现的次数</li><li>&quot;degree&quot;, 图中连接数（实体节点的度数）</li><li>&quot;x&quot;, 可视化布局的X坐标</li><li>&quot;y&quot;, 可视化布局的Y坐标</li></ul></li><li>finalize_relationships <ul><li>&quot;id&quot;, 关系ID</li><li>&quot;human_readable_id&quot;, 可读的有序关系ID</li><li>&quot;source&quot;, 关系的起始实体ID</li><li>&quot;target&quot;, 关系的结束实体ID</li><li>&quot;description&quot;, 关系的描述</li><li>&quot;weight&quot;, 关系权重（连接强度）</li><li>&quot;combined_degree&quot;, 综合连接度（两端节点degree之和，反映关系重要性）</li><li>&quot;text_unit_ids&quot;, 分块ID列表</li></ul></li></ul></li><li>更新entities.parquet和relationships.parquet文件</li></ul><h3 id="_4-10-执行pipeline-create-communities-将节点聚类成社区" tabindex="-1"><a class="header-anchor" href="#_4-10-执行pipeline-create-communities-将节点聚类成社区" aria-hidden="true">#</a> 4.10. 执行pipeline : create_communities（将节点聚类成社区）</h3><ul><li>加载entities.parquet和relationships.parquet文件</li><li>create_communities</li><li>生成communities.parquet <ul><li>&quot;id&quot;, 社区ID</li><li>&quot;human_readable_id&quot;, 可读有序社区ID</li><li>&quot;community&quot;, 社区父子关系链接ID</li><li>&quot;level&quot;, 当前层级</li><li>&quot;parent&quot;, 父级</li><li>&quot;children&quot;, 子级</li><li>&quot;title&quot;, 社区名称</li><li>&quot;entity_ids&quot;, 实体ID列表</li><li>&quot;relationship_ids&quot;, 关系ID列表</li><li>&quot;text_unit_ids&quot;, 分块ID列表</li><li>&quot;period&quot;, 时间戳</li><li>&quot;size&quot;, 社区大小</li></ul></li></ul><h3 id="_4-11-执行pipeline-create-final-text-units" tabindex="-1"><a class="header-anchor" href="#_4-11-执行pipeline-create-final-text-units" aria-hidden="true">#</a> 4.11. 执行pipeline : create_final_text_units</h3><ul><li>加载text_units.parquet、entities.parquet和relationships.parquet文件</li><li>create_final_text_units</li><li>更新text_units.parquet <ul><li>&#39;id&#39; : 分块ID</li><li>&#39;human_readable_id&#39; : 可读有序分块ID</li><li>&#39;text&#39; : 分块文本</li><li>&#39;document_ids&#39; : 所属文件ID</li><li>&#39;n_tokens&#39; : token数</li><li>&#39;entity_ids&#39; : 实体ID列表</li><li>&#39;relationship_ids&#39; : 关系ID列表</li></ul></li></ul><h3 id="_4-12-执行pipeline-create-community-reports-生成社区报告" tabindex="-1"><a class="header-anchor" href="#_4-12-执行pipeline-create-community-reports-生成社区报告" aria-hidden="true">#</a> 4.12. 执行pipeline : create_community_reports（生成社区报告）</h3><ul><li>加载entities.parquet、relationships.parquet和communities.parquet</li><li>create_community_reports</li><li>生成community_reports.parquet <ul><li>&quot;id&quot;,</li><li>&quot;human_readable_id&quot;,</li><li>&quot;community&quot;,</li><li>&quot;level&quot;,</li><li>&quot;parent&quot;,</li><li>&quot;children&quot;,</li><li>&quot;title&quot;,</li><li>&quot;summary&quot;,</li><li>&quot;full_content&quot;,</li><li>&quot;rank&quot;,</li><li>&quot;rank_explanation&quot;,</li><li>&quot;findings&quot;,</li><li>&quot;full_content_json&quot;,</li><li>&quot;period&quot;,</li><li>&quot;size&quot;,</li></ul></li></ul><h3 id="_4-13-执行pipeline-generate-text-embeddings" tabindex="-1"><a class="header-anchor" href="#_4-13-执行pipeline-generate-text-embeddings" aria-hidden="true">#</a> 4.13. 执行pipeline : generate_text_embeddings</h3><ul><li>加载documents.parquet文件、text_unit.parquet、entities.parquet、relationships.parquet和community_reports.parquet</li><li>generate_text_embeddings</li></ul><h2 id="_5-query-源码逻辑" tabindex="-1"><a class="header-anchor" href="#_5-query-源码逻辑" aria-hidden="true">#</a> 5. query 源码逻辑</h2><p>四种搜索方式</p><ul><li>local</li><li>global</li><li>drift</li><li>basic</li></ul><h3 id="_5-1-local-search-搜索实体或关系的细节问题" tabindex="-1"><a class="header-anchor" href="#_5-1-local-search-搜索实体或关系的细节问题" aria-hidden="true">#</a> 5.1. local_search -&gt; 搜索实体或关系的细节问题</h3><figure><img src="`+a+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_5-2-global-search-搜索社区的广泛问题" tabindex="-1"><a class="header-anchor" href="#_5-2-global-search-搜索社区的广泛问题" aria-hidden="true">#</a> 5.2. global_search -&gt; 搜索社区的广泛问题</h3><figure><img src="'+s+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_5-3-drift-search-结合了local-search和global-search" tabindex="-1"><a class="header-anchor" href="#_5-3-drift-search-结合了local-search和global-search" aria-hidden="true">#</a> 5.3. drift_search -&gt; 结合了local_search和global_search</h3><p>https://microsoft.github.io/graphrag/query/drift_search/</p><figure><img src="'+r+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure>',60);function f(y,k){const n=u("ExternalLinkIcon");return o(),c("div",null,[p,b,v(" more "),q,i("p",null,[e("Github: "),i("a",h,[e("https://github.com/microsoft/graphrag"),l(n)])]),i("p",null,[e("Get Started: "),i("a",g,[e("https://microsoft.github.io/graphrag/get_started/"),l(n)])]),x])}const z=d(_,[["render",f],["__file","021_microsoft_graphrag.html.vue"]]);export{z as default};
