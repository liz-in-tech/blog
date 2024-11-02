import{_ as n,a as o,b as r,c as l,d as s,e as d,f as h}from"./013_accuracy_score-YT8TJEN6.js";import{_ as c}from"./plugin-vue_export-helper-x3n3nnut.js";import{r as u,o as m,c as p,f as g,a as e,b as i,d as a,e as f}from"./app-YkV2VZ8k.js";const w={},b=e("h1",{id:"best-practices-for-optimizing-llms-prompt-engineering-rag-and-fine-tuning",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#best-practices-for-optimizing-llms-prompt-engineering-rag-and-fine-tuning","aria-hidden":"true"},"#"),i(" Best Practices for Optimizing LLMs (Prompt Engineering, RAG and Fine-tuning)")],-1),x=f('<h2 id="_1-the-challenges-of-optimizing-llms" tabindex="-1"><a class="header-anchor" href="#_1-the-challenges-of-optimizing-llms" aria-hidden="true">#</a> 1. The Challenges of Optimizing LLMs</h2><ul><li>Extracting signal (meaningful information) from the noise (irrelevant information) is not easy</li><li>Performance can be abstract and difficult to measure</li><li>It’s not clear when to use what kind of method to optimize LLMs.</li></ul><h2 id="_2-the-optimization-strategies" tabindex="-1"><a class="header-anchor" href="#_2-the-optimization-strategies" aria-hidden="true">#</a> 2. The Optimization Strategies</h2><p>Optimizing LLMs can be thought of as a two-axis problem</p><figure><img src="'+n+'" alt="The Optimization Strategies" tabindex="0" loading="lazy"><figcaption>The Optimization Strategies</figcaption></figure><ul><li>First to Optimize Prompt</li><li>Context Optimization (What the model needs to know?) =&gt; RAG</li><li>LLM Optimization (How the model needs to act ?) =&gt; Fine-Tuning</li></ul><h2 id="_3-typical-optimization-pipeline" tabindex="-1"><a class="header-anchor" href="#_3-typical-optimization-pipeline" aria-hidden="true">#</a> 3. Typical Optimization Pipeline</h2><figure><img src="'+o+'" alt="Typical Optimization Pipeline" tabindex="0" loading="lazy"><figcaption>Typical Optimization Pipeline</figcaption></figure><ul><li>1.Start with a prompt</li><li>2.Get proper evaluation metrics</li><li>3.Once evaluation metrics are set, figure out the baseline</li><li>4.Once the baseline is known, add more few shot examples <ul><li>This is required to guide the model, about how the customer wants the model to act</li></ul></li><li>5.If adding a few shots leads to an increase in performance, then follow the RAG process</li><li>6.After RAG is done, the model is getting the context right, but it is not producing the output in a required format, so then fine-tuning approach is taken.</li><li>7.It can be a possibility, that retrieval might not be as good as one wants, then go back to RAG and optimize the RAG again. <ul><li>For example, in RAG maybe add Hypothetical Document Embeddings (HyDE) retrieval + fast-checking step.</li><li>Then fine-tune the model again, with these new examples added as context via RAG.</li></ul></li></ul><h2 id="_4-comparison-of-optimisation-approaches" tabindex="-1"><a class="header-anchor" href="#_4-comparison-of-optimisation-approaches" aria-hidden="true">#</a> 4. Comparison of Optimisation Approaches</h2><table><thead><tr><th></th><th>Prompt Engineering</th><th>RAG</th><th>Fine-tuning</th></tr></thead><tbody><tr><td>Reducing token usage</td><td>✕</td><td>✕</td><td>✓</td></tr><tr><td>Introducing new information</td><td>✕</td><td>✓</td><td>✓</td></tr><tr><td>Testing and learning early</td><td>✓</td><td>✕</td><td>✕</td></tr><tr><td>Reducing hallucinations</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>Improving efficiency</td><td>✕</td><td>✕</td><td>✓</td></tr></tbody></table><h2 id="_5-optimization-via-prompt-engineering" tabindex="-1"><a class="header-anchor" href="#_5-optimization-via-prompt-engineering" aria-hidden="true">#</a> 5. Optimization via Prompt Engineering</h2><ul><li>Write clear instructions</li><li>Split complex tasks into simpler subtasks</li><li>Give llm time to “think”</li><li>Test changes systematically</li><li>Providing Examples</li><li>Using external Tools</li></ul><h2 id="_6-how-to-evaluate-rag" tabindex="-1"><a class="header-anchor" href="#_6-how-to-evaluate-rag" aria-hidden="true">#</a> 6. How to Evaluate RAG</h2><figure><img src="'+r+'" alt="RAG Metrics" tabindex="0" loading="lazy"><figcaption>RAG Metrics</figcaption></figure><p>RAG can be evaluated using 4 metrics. Two of the metrics are inclined towards LLMs and two towards context.</p><h3 id="_6-1-llm-related-metrics" tabindex="-1"><a class="header-anchor" href="#_6-1-llm-related-metrics" aria-hidden="true">#</a> 6.1. LLM-related Metrics</h3><ul><li>Faithfulness <ul><li>Takes the answer, chunk it, and tries to reconcile the answer with the facts</li><li>If the answer can’t be reconciled as a fact, then the answer is hallucinated</li></ul></li><li>Answer Relevancy <ul><li>Let’s say that the model is provided with a lot of context, now the model makes use of that context and provides the answer, but the answer is nothing near to what the user wanted or originally asked</li><li>Thus, this metric looks at the relevancy of the answer provided by the model.</li></ul></li></ul><h3 id="_6-2-context-related-metrics" tabindex="-1"><a class="header-anchor" href="#_6-2-context-related-metrics" aria-hidden="true">#</a> 6.2. Context-related Metrics</h3><ul><li>Context Precision <ul><li>Most useful from the customer perspective, as there can be scenarios where model accuracy is high, but context precision is low.</li><li>Classic RAG scenario can be thought of as being able to put more and more context in the context window, however as the model gets more context, the model hallucinations might increase (Refer paper: Lost in the Middle: How Language Models Use Long Contexts)</li><li>Thus, context precision evaluates the signal-to-noise ratio of the retrieved content. It takes the content log and compares it with the answer, and figures out whether the retrieved content matches the “to be answer”.</li></ul></li><li>Context Recall <ul><li>Can the model retrieve all the relevant information required to answer the question?</li><li>Does the search which is put at the top by the model is answering the question?</li><li>context recalls tell if the search needs to be optimized, may need to add reranking, fine-tune embeddings, or may be different embeddings are needed to surface more relevant content.</li></ul></li></ul><h2 id="_7-optimization-via-fine-tuning" tabindex="-1"><a class="header-anchor" href="#_7-optimization-via-fine-tuning" aria-hidden="true">#</a> 7. Optimization via Fine-Tuning</h2><p>Fine-tuning is a process of continuing the training of a model on a smaller domain-specific dataset to optimize the model for a specific task.</p><p>Why Fine-Tune?</p><ul><li>Reducing token usage <ul><li>Reduced limitations on context windows and exposed to much more data</li></ul></li><li>Improving model efficiency <ul><li>It is observed that complex prompting techniques are not required to reach the desired level of performance, once a model is fine-tuned</li><li>No need to provide a complex set of instructions, explicit schemas, in-context examples</li><li>Fewer prompt tokens are needed per request, thus making the interaction cheaper and leading to a quicker response</li><li>Knowledge distillation: knowledge distillation via fine-tuning say from a very large model like GPT-4 turbo, to a smaller model like GPT-3.5 which is much cheaper (cost and latency-wise)</li></ul></li></ul><h2 id="_8-best-practices-in-fine-tuning" tabindex="-1"><a class="header-anchor" href="#_8-best-practices-in-fine-tuning" aria-hidden="true">#</a> 8. Best Practices in Fine-Tuning</h2><ul><li>1.Start with prompt engineering and a few shot learning (FSL) <ul><li>These are low-investment techniques to quickly iterate and validate a use-case</li><li>These can give intuition about how LLMs operate, and how they work on a specific problem.</li></ul></li><li>2.Establish a baseline <ul><li>Ensure a performance baseline to compare the fine-tuned model to Understand the failure cases of the model</li><li>Understand the exact target to be achieved via fine-tuning</li></ul></li><li>3.Start small, focus on quality <ul><li>Datasets are difficult to build, so start small and invest intentionally</li><li>Fine-tune a model on a smaller dataset, look at its output, see what area it struggles in, and then target those areas with new data</li><li>Optimize for fewer high-quality training examples.</li></ul></li><li>4.Devise proper evaluation strategies <ul><li>Expert humans to look at the output and rank them on some scale</li><li>Generate output from different models, and get the model rank them for you. For example, use GPT-4 to rank the output of some open source model.</li><li>Train the model, evaluate it, and deploy it to production. Then, collect the samples from it in production, use that to build a new dataset, downsample that dataset, curate it, and then fine-tune again on that dataset.</li></ul></li></ul><h2 id="_9-best-practices-in-fine-tuning-rag" tabindex="-1"><a class="header-anchor" href="#_9-best-practices-in-fine-tuning-rag" aria-hidden="true">#</a> 9. Best Practices in Fine-Tuning + RAG</h2><ul><li>1.Fine-tune the model to understand the complex instructions <ul><li>It will eliminate the need to provide complex few-shot examples to the model at a sample time</li><li>It will also minimize the prompt-engineering tokens, leading to more space for retrieved context.</li></ul></li><li>2.Next, use RAG to inject relevant knowledge into the context needed to be maximized, but do not over-saturate the context</li></ul><h2 id="_10-openai-rag-use-case" tabindex="-1"><a class="header-anchor" href="#_10-openai-rag-use-case" aria-hidden="true">#</a> 10. OpenAI RAG Use Case</h2><p>One of the OpenAI customers had a pipeline with 2 knowledge bases and an LLM. The job was to take the user question, decide which knowledge base to use, fire a query, and use one of the knowledge bases to answer a question.</p><figure><img src="'+l+'" alt="OpenAI RAG Use Case" tabindex="0" loading="lazy"><figcaption>OpenAI RAG Use Case</figcaption></figure><h3 id="_10-1-experiments-that-didn-t-work" tabindex="-1"><a class="header-anchor" href="#_10-1-experiments-that-didn-t-work" aria-hidden="true">#</a> 10.1. Experiments that didn’t work</h3><ul><li>Retrieval with cosine similarity (gave 45 % accuracy)</li><li>HyDE retrieval was tried but didn’t pass through for production.</li><li>Fine-tuning the embeddings worked well from the accuracy perspective, but was slow and expensive, so they discarded it for non-functional reasons.</li></ul><h3 id="_10-2-experiments-which-worked" tabindex="-1"><a class="header-anchor" href="#_10-2-experiments-which-worked" aria-hidden="true">#</a> 10.2. Experiments which worked</h3><ul><li>Chunk / embedding experiments <ul><li>Trying different size chunks of information and embedding different bits of content gave a 20% accuracy bump to 65%</li></ul></li><li>Reranking <ul><li>Applied cross-encoder to rerank or rules-based ranking</li><li>rerank <ul><li>cohere -&gt; rerank (good) https://docs.cohere.com/reference/rerank</li><li>bje reranker （open source）https://huggingface.co/collections/BAAI/bge-66797a74476eb1f085c7446d</li><li>jina</li></ul></li><li>Cross Encoder：https://www.cnblogs.com/huggingface/p/18010292</li></ul></li><li>Classification Step <ul><li>Classify which two domains (two knowledge bases), the question could belong to and give extra metadata in the prompt to help it decide further</li></ul></li><li>To reach accuracy levels of 98%, the following trials were successful <ul><li>Further prompt engineering to engineer the prompt a lot better</li><li>Looked at the category of questions that gave wrong answers, and tools were introduced like giving access to SQL databases to pull the answers from.</li><li>Query Expansion: where someone has asked 3 questions, in 1 prompt, then parses them into a list of queries, executes them in parallel, brings back the results, and synthesizes them into 1 result.</li><li>Fine-tuning was not used here, which showed that the problem was context-related.</li><li>Fine-tuning could have led to the waste of computational resources in a context-related setting.</li></ul></li></ul><h2 id="_11-openai-fine-tuning-rag-use-case" tabindex="-1"><a class="header-anchor" href="#_11-openai-fine-tuning-rag-use-case" aria-hidden="true">#</a> 11. OpenAI Fine-Tuning + RAG Use Case</h2><p>Use case description: Given a natural language question and a database schema, can the model produce a semantically correct SQL query?</p><figure><img src="'+s+'" alt="OpenAI Fine-Tuning + RAG Use Case" tabindex="0" loading="lazy"><figcaption>OpenAI Fine-Tuning + RAG Use Case</figcaption></figure><p>The approach was first followed with the GPT 3.5 Turbo Model</p><figure><img src="'+d+'" alt="Overall Score" tabindex="0" loading="lazy"><figcaption>Overall Score</figcaption></figure><figure><img src="'+h+'" alt="Accuracy Score" tabindex="0" loading="lazy"><figcaption>Accuracy Score</figcaption></figure><ul><li>1.First, the performance was squeezed from the baseline model via prompt engineering ( 69%). A few shot examples were also added, which led to some improvement leading to the use of RAG</li><li>2.with the RAG question only, there was a 3% performance bump</li><li>3.Then using the answers, hypothetical document embedding, there was a further 5% improvement. <ul><li>Here just using hypothetical questions to search, rather than the actual input questions led to improvement</li></ul></li><li>4.Increased examples from n=3 to n=5 lead to further improvement to 80%</li><li>5.However the target was to reach 84%.</li><li>6.Fine-tuning was employed (done at ScaleAI) <ul><li>ScaleAI fine-tuned GPT-4 with a simple prompt engineering technique of reducing the Schema gave almost 82%</li><li>They used RAG along with fine-tuning to dynamically inject a few examples into the context window, which led to 83.5% accuracy score.</li></ul></li><li>7.Thus, simple fine-tuning + RAG combined with simple prompt engineering brought the model accuracy to 83.5%</li></ul><h2 id="_12-reference" tabindex="-1"><a class="header-anchor" href="#_12-reference" aria-hidden="true">#</a> 12. Reference</h2>',43),v={href:"https://www.youtube.com/watch?v=ahnGLM-RC1Y",target:"_blank",rel:"noopener noreferrer"},_=e("br",null,null,-1),y={href:"https://medium.com/@luvverma2011/optimizing-llms-best-practices-prompt-engineering-rag-and-fine-tuning-8def58af8dcc",target:"_blank",rel:"noopener noreferrer"};function k(z,A){const t=u("ExternalLinkIcon");return m(),p("div",null,[b,g(" more "),x,e("p",null,[e("a",v,[i("OpenAI:A Survey of Techniques for Maximizing LLM Performance"),a(t)]),_,e("a",y,[i("Optimizing LLMs: Best Practices"),a(t)])])])}const L=c(w,[["render",k],["__file","013_optimizing_llm.html.vue"]]);export{L as default};