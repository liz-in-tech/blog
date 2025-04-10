import{_ as l,a,b as r,c as s}from"./021_graphrag_drift_search-N3Np96tC.js";import{_ as d}from"./plugin-vue_export-helper-x3n3nnut.js";import{r as o,o as u,c,f as m,a as e,b as i,d as t,e as v}from"./app-kXW_Y_m4.js";const p={},_=e("h1",{id:"microsoft-graphrag-source-code-interpretation",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#microsoft-graphrag-source-code-interpretation","aria-hidden":"true"},"#"),i(" Microsoft GraphRAG Source Code Interpretation")],-1),h=e("ul",null,[e("li",null,"Source Code Entry Point"),e("li",null,"init Source Code Logic"),e("li",null,"index Source Code Logic"),e("li",null,"query Source Code Logic")],-1),b=e("h2",{id:"_1-official-documentation",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#_1-official-documentation","aria-hidden":"true"},"#"),i(" 1. Official Documentation")],-1),g={href:"https://github.com/microsoft/graphrag",target:"_blank",rel:"noopener noreferrer"},q={href:"https://microsoft.github.io/graphrag/get_started/",target:"_blank",rel:"noopener noreferrer"},f=v(`<p>Paper: https://arxiv.org/pdf/2404.16130</p><h2 id="_2-source-code-entry-point" tabindex="-1"><a class="header-anchor" href="#_2-source-code-entry-point" aria-hidden="true">#</a> 2. Source Code Entry Point</h2><blockquote><p>graphrag.cli.main.py</p></blockquote><p>Five functions correspond to five commands:</p><ul><li>init =&gt; def _initialize_cli() <ul><li>graphrag init --root ./ragtest</li></ul></li><li>index =&gt; def _index_cli() <ul><li>graphrag index --root ./ragtest</li></ul></li><li>query =&gt; def _query_cli() <ul><li>global search <ul><li>graphrag query --root ./ragtest --method global --query &quot;What are the top themes in this story?&quot;</li></ul></li><li>local search <ul><li>graphrag query --root ./ragtest --method local --query &quot;Who is Scrooge and what are his main relationships?&quot;</li></ul></li></ul></li><li>update =&gt; def _update_cli() <ul><li>graphrag update</li></ul></li><li>prompt-tune =&gt; def _prompt_tune_cli() <ul><li>graphrag prompt-tune</li></ul></li></ul><p>Principle: Uses Python&#39;s Typer library to create a command-line interface (CLI). Take index as an example:</p><ul><li>@app.command(&quot;index&quot;) is a decorator syntax used to register the _index_cli function below as a CLI command, creating a subcommand named &quot;index&quot;.</li><li>When the user runs &quot;graphrag index&quot; on the command line, the decorated def _index_cli() function is called.</li><li>Typer automatically converts function parameters into command-line options and handles parameter parsing and validation.</li></ul><h3 id="_2-1-init" tabindex="-1"><a class="header-anchor" href="#_2-1-init" aria-hidden="true">#</a> 2.1. init</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>@app.command(&quot;init&quot;)
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_2-3-query" tabindex="-1"><a class="header-anchor" href="#_2-3-query" aria-hidden="true">#</a> 2.3. query</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>@app.command(&quot;query&quot;)
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="_3-init-source-code-logic" tabindex="-1"><a class="header-anchor" href="#_3-init-source-code-logic" aria-hidden="true">#</a> 3. init Source Code Logic</h2><ul><li>Generate initialization files in the specified root directory <ul><li>Generate configuration file settings.yaml</li><li>Generate environment variable file .env</li><li>Generate multiple xxx_prompt.txt files in the prompts path of the root directory</li></ul></li></ul><h2 id="_4-index-source-code-logic" tabindex="-1"><a class="header-anchor" href="#_4-index-source-code-logic" aria-hidden="true">#</a> 4. index Source Code Logic</h2><figure><img src="`+l+`" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_4-1-load-configuration-file" tabindex="-1"><a class="header-anchor" href="#_4-1-load-configuration-file" aria-hidden="true">#</a> 4.1. Load Configuration File</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.cli.index.py -&gt; def index_cli()

config = load_config(root_dir, config_filepath, cli_overrides)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-2-build-index" tabindex="-1"><a class="header-anchor" href="#_4-2-build-index" aria-hidden="true">#</a> 4.2. Build Index</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.api.index.py -&gt; def build_index()

async def build_index(
    config: GraphRagConfig,
    method: IndexingMethod = IndexingMethod.Standard,
    is_update_run: bool = False,
    memory_profile: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    progress_logger: ProgressLogger | None = None,
) -&gt; list[PipelineRunResult]:

    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config, method)

    # Execute pipeline
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-3-create-pipeline" tabindex="-1"><a class="header-anchor" href="#_4-3-create-pipeline" aria-hidden="true">#</a> 4.3. Create Pipeline</h3><p>The parameter method determines the type of pipeline:</p><ul><li>Standard Pipeline: IndexingMethod.Standard =&gt; full LLM <ul><li>&quot;create_base_text_units&quot;,</li><li>&quot;create_final_documents&quot;,</li><li>&quot;extract_graph&quot;,</li><li>&quot;finalize_graph&quot;,</li><li>*([&quot;extract_covariates&quot;] if config.extract_claims.enabled else []),</li><li>&quot;create_communities&quot;,</li><li>&quot;create_final_text_units&quot;,</li><li>&quot;create_community_reports&quot;,</li><li>&quot;generate_text_embeddings&quot;,</li></ul></li><li>Fast Pipeline: IndexingMethod.Fast =&gt; NLP + LLM <ul><li>&quot;create_base_text_units&quot;,</li><li>&quot;create_final_documents&quot;,</li><li>&quot;extract_graph_nlp&quot;,</li><li>&quot;prune_graph&quot;,</li><li>&quot;finalize_graph&quot;,</li><li>&quot;create_communities&quot;,</li><li>&quot;create_final_text_units&quot;,</li><li>&quot;create_community_reports_text&quot;,</li><li>&quot;generate_text_embeddings&quot;,</li></ul></li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.workflows.factory.py -&gt; def create_pipeline()

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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-4-load-input-dataset" tabindex="-1"><a class="header-anchor" href="#_4-4-load-input-dataset" aria-hidden="true">#</a> 4.4. Load Input Dataset</h3><ul><li>Load files in the input folder of the root directory</li><li>Currently supports txt and csv file types</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.run.run_pipeline.py -&gt; def run_pipeline()

dataset = await create_input(config.input, logger, root_dir)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-5-generate-documents-parquet-file" tabindex="-1"><a class="header-anchor" href="#_4-5-generate-documents-parquet-file" aria-hidden="true">#</a> 4.5. Generate documents.parquet File</h3><ul><li>Rows: Each file in the input folder is a row</li><li>Columns: <ul><li>&#39;text&#39; : Text content in the document</li><li>&#39;id&#39; : Document ID</li><li>&#39;title&#39; : Document name</li><li>&#39;creation_date&#39; : Creation date</li></ul></li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code># graphrag.index.run.run_pipeline.py -&gt; def _run_pipeline()

await write_table_to_storage(dataset, &quot;documents&quot;, context.storage)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-6-execute-pipeline-create-base-text-units-document-chunking" tabindex="-1"><a class="header-anchor" href="#_4-6-execute-pipeline-create-base-text-units-document-chunking" aria-hidden="true">#</a> 4.6. Execute Pipeline: create_base_text_units (Document Chunking)</h3><ul><li>Load documents.parquet file</li><li>Generate text_unit.parquet file <ul><li>Rows: Each row in documents.parquet is chunked, each chunk is a row</li><li>Columns: <ul><li>&#39;id&#39; : Chunk ID</li><li>&#39;text&#39; : Chunk text</li><li>&#39;document_ids&#39; : Associated document ID</li><li>&#39;n_tokens&#39; : Number of tokens</li></ul></li></ul></li></ul><h3 id="_4-7-execute-pipeline-create-final-documents-aggregate-chunk-id-list-to-documents-parquet-file" tabindex="-1"><a class="header-anchor" href="#_4-7-execute-pipeline-create-final-documents-aggregate-chunk-id-list-to-documents-parquet-file" aria-hidden="true">#</a> 4.7. Execute Pipeline: create_final_documents (Aggregate Chunk ID List to documents.parquet File)</h3><ul><li>Load documents.parquet and text_unit.parquet files</li><li>create_final_documents <ul><li>Merge documents.parquet and text_unit.parquet files with document ID as the association for inner join</li><li>Group by document ID, aggregate chunk IDs as text_unit_ids column</li></ul></li><li>Update documents.parquet file <ul><li>Rows: Each file in the input folder is a row</li><li>Columns: <ul><li>&#39;id&#39; : Document ID (unique unordered ID)</li><li>&#39;human_readable_id&#39; : Readable sequential ID (sequentially numbered ID)</li><li>&#39;text&#39; : Text content in the document</li><li>&#39;text_unit_id&#39; : List of chunk IDs</li><li>&#39;creation_date&#39; : Creation date</li></ul></li></ul></li></ul><h3 id="_4-8-execute-pipeline-extract-graph" tabindex="-1"><a class="header-anchor" href="#_4-8-execute-pipeline-extract-graph" aria-hidden="true">#</a> 4.8. Execute Pipeline: extract_graph</h3><ul><li>Load text_unit.parquet file</li><li>extract_graph (see code below) <ul><li>load_llm</li><li>run_extract_graph to extract entities and relationships (see code below)</li><li>run_summarize_descriptions to generate summary descriptions (see code below)</li><li>return (entities, relationships) <ul><li>entities <ul><li>&quot;title&quot;, Entity name</li><li>&quot;type&quot;, Entity type</li><li>&quot;description&quot;, Entity description</li><li>&quot;text_unit_ids&quot;, List of chunk IDs</li><li>&quot;frequency&quot;, Frequency of entity occurrence</li></ul></li><li>relationships <ul><li>&quot;source&quot;, Starting entity ID of the relationship</li><li>&quot;target&quot;, Ending entity ID of the relationship</li><li>&quot;description&quot;, Description of the relationship</li><li>&quot;weight&quot;, Weight of the relationship (strength of connection)</li><li>&quot;text_unit_ids&quot;, List of chunk IDs</li></ul></li></ul></li></ul></li><li>Generate entities.parquet and relationships.parquet files</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>graphrag.index.workflows.extract_graph.py -&gt; def extract_graph()

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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_4-9-execute-pipeline-finalize-graph" tabindex="-1"><a class="header-anchor" href="#_4-9-execute-pipeline-finalize-graph" aria-hidden="true">#</a> 4.9. Execute Pipeline: finalize_graph</h3><ul><li>Load entities.parquet and relationships.parquet files</li><li>finalize_graph <ul><li>finalize_entities <ul><li>&quot;id&quot;, Entity ID</li><li>&quot;human_readable_id&quot;, Readable ordered entity ID</li><li>&quot;title&quot;, Entity name</li><li>&quot;type&quot;, Entity type</li><li>&quot;description&quot;, Entity description</li><li>&quot;text_unit_ids&quot;, List of chunk IDs</li><li>&quot;frequency&quot;, Frequency of entity occurrence</li><li>&quot;degree&quot;, Number of connections in the graph (degree of the entity node)</li><li>&quot;x&quot;, X-coordinate of the visualization layout</li><li>&quot;y&quot;, Y-coordinate of the visualization layout</li></ul></li><li>finalize_relationships <ul><li>&quot;id&quot;, Relationship ID</li><li>&quot;human_readable_id&quot;, Readable ordered relationship ID</li><li>&quot;source&quot;, Starting entity ID of the relationship</li><li>&quot;target&quot;, Ending entity ID of the relationship</li><li>&quot;description&quot;, Description of the relationship</li><li>&quot;weight&quot;, Weight of the relationship (strength of connection)</li><li>&quot;combined_degree&quot;, Combined degree (sum of degrees of both nodes, reflecting the importance of the relationship)</li><li>&quot;text_unit_ids&quot;, List of chunk IDs</li></ul></li></ul></li><li>Update entities.parquet and relationships.parquet files</li></ul><h3 id="_4-10-execute-pipeline-create-communities-cluster-nodes-into-communities" tabindex="-1"><a class="header-anchor" href="#_4-10-execute-pipeline-create-communities-cluster-nodes-into-communities" aria-hidden="true">#</a> 4.10. Execute Pipeline: create_communities (Cluster Nodes into Communities)</h3><ul><li>Load entities.parquet and relationships.parquet files</li><li>create_communities</li><li>Generate communities.parquet <ul><li>&quot;id&quot;, Community ID</li><li>&quot;human_readable_id&quot;, Readable ordered community ID</li><li>&quot;community&quot;, Community parent-child relationship link ID</li><li>&quot;level&quot;, Current level</li><li>&quot;parent&quot;, Parent</li><li>&quot;children&quot;, Children</li><li>&quot;title&quot;, Community name</li><li>&quot;entity_ids&quot;, List of entity IDs</li><li>&quot;relationship_ids&quot;, List of relationship IDs</li><li>&quot;text_unit_ids&quot;, List of chunk IDs</li><li>&quot;period&quot;, Timestamp</li><li>&quot;size&quot;, Community size</li></ul></li></ul><h3 id="_4-11-execute-pipeline-create-final-text-units" tabindex="-1"><a class="header-anchor" href="#_4-11-execute-pipeline-create-final-text-units" aria-hidden="true">#</a> 4.11. Execute Pipeline: create_final_text_units</h3><ul><li>Load text_units.parquet, entities.parquet, and relationships.parquet files</li><li>create_final_text_units</li><li>Update text_units.parquet <ul><li>&#39;id&#39; : Chunk ID</li><li>&#39;human_readable_id&#39; : Readable ordered chunk ID</li><li>&#39;text&#39; : Chunk text</li><li>&#39;document_ids&#39; : Associated document ID</li><li>&#39;n_tokens&#39; : Number of tokens</li><li>&#39;entity_ids&#39; : List of entity IDs</li><li>&#39;relationship_ids&#39; : List of relationship IDs</li></ul></li></ul><h3 id="_4-12-execute-pipeline-create-community-reports-generate-community-reports" tabindex="-1"><a class="header-anchor" href="#_4-12-execute-pipeline-create-community-reports-generate-community-reports" aria-hidden="true">#</a> 4.12. Execute Pipeline: create_community_reports (Generate Community Reports)</h3><ul><li>Load entities.parquet, relationships.parquet, and communities.parquet</li><li>create_community_reports</li><li>Generate community_reports.parquet <ul><li>&quot;id&quot;,</li><li>&quot;human_readable_id&quot;,</li><li>&quot;community&quot;,</li><li>&quot;level&quot;,</li><li>&quot;parent&quot;,</li><li>&quot;children&quot;,</li><li>&quot;title&quot;,</li><li>&quot;summary&quot;,</li><li>&quot;full_content&quot;,</li><li>&quot;rank&quot;,</li><li>&quot;rank_explanation&quot;,</li><li>&quot;findings&quot;,</li><li>&quot;full_content_json&quot;,</li><li>&quot;period&quot;,</li><li>&quot;size&quot;,</li></ul></li></ul><h3 id="_4-13-execute-pipeline-generate-text-embeddings" tabindex="-1"><a class="header-anchor" href="#_4-13-execute-pipeline-generate-text-embeddings" aria-hidden="true">#</a> 4.13. Execute Pipeline: generate_text_embeddings</h3><ul><li>Load documents.parquet file, text_unit.parquet, entities.parquet, relationships.parquet, and community_reports.parquet</li><li>generate_text_embeddings</li></ul><h2 id="_5-query-source-code-logic" tabindex="-1"><a class="header-anchor" href="#_5-query-source-code-logic" aria-hidden="true">#</a> 5. query Source Code Logic</h2><p>Four search methods:</p><ul><li>local</li><li>global</li><li>drift</li><li>basic</li></ul><h3 id="_5-1-local-search-search-for-detailed-questions-about-entities-or-relationships" tabindex="-1"><a class="header-anchor" href="#_5-1-local-search-search-for-detailed-questions-about-entities-or-relationships" aria-hidden="true">#</a> 5.1. local_search -&gt; Search for detailed questions about entities or relationships</h3><figure><img src="`+a+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_5-2-global-search-search-for-broad-questions-about-communities" tabindex="-1"><a class="header-anchor" href="#_5-2-global-search-search-for-broad-questions-about-communities" aria-hidden="true">#</a> 5.2. global_search -&gt; Search for broad questions about communities</h3><figure><img src="'+r+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_5-3-drift-search-combines-local-search-and-global-search" tabindex="-1"><a class="header-anchor" href="#_5-3-drift-search-combines-local-search-and-global-search" aria-hidden="true">#</a> 5.3. drift_search -&gt; Combines local_search and global_search</h3><p>https://microsoft.github.io/graphrag/query/drift_search/</p><figure><img src="'+s+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure>',60);function x(y,k){const n=o("ExternalLinkIcon");return u(),c("div",null,[_,h,m(" more "),b,e("p",null,[i("GitHub: "),e("a",g,[i("https://github.com/microsoft/graphrag"),t(n)])]),e("p",null,[i("Get Started: "),e("a",q,[i("https://microsoft.github.io/graphrag/get_started/"),t(n)])]),f])}const D=d(p,[["render",x],["__file","021_microsoft_graphrag.html.vue"]]);export{D as default};
