const e=JSON.parse('{"key":"v-1c829e52","path":"/posts/LLM/transformer.html","title":"Transformer Source Code Exploration","lang":"en-US","frontmatter":{"icon":"lightbulb","date":"2024-05-24T00:00:00.000Z","sticky":true,"star":true,"category":["LLM"],"tag":["LLM"],"description":"Transformer Source Code Exploration About Transformer Overall Architecture Hyperparameters Tensor Dimensionality Transformation Number of Trainable Parameters Source Code","head":[["link",{"rel":"alternate","hreflang":"zh-cn","href":"https://liz-starfield.github.io/blog/zh/posts/LLM/transformer.html"}],["meta",{"property":"og:url","content":"https://liz-starfield.github.io/blog/posts/LLM/transformer.html"}],["meta",{"property":"og:site_name","content":"Liz"}],["meta",{"property":"og:title","content":"Transformer Source Code Exploration"}],["meta",{"property":"og:description","content":"Transformer Source Code Exploration About Transformer Overall Architecture Hyperparameters Tensor Dimensionality Transformation Number of Trainable Parameters Source Code"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"og:locale:alternate","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2024-06-03T01:33:06.000Z"}],["meta",{"property":"article:author","content":"Liz"}],["meta",{"property":"article:tag","content":"LLM"}],["meta",{"property":"article:published_time","content":"2024-05-24T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2024-06-03T01:33:06.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"Transformer Source Code Exploration\\",\\"image\\":[\\"\\"],\\"datePublished\\":\\"2024-05-24T00:00:00.000Z\\",\\"dateModified\\":\\"2024-06-03T01:33:06.000Z\\",\\"author\\":[{\\"@type\\":\\"Person\\",\\"name\\":\\"Liz\\",\\"url\\":\\"https://github.com/liz-starfield\\"}]}"]]},"headers":[{"level":2,"title":"1. About","slug":"_1-about","link":"#_1-about","children":[]},{"level":2,"title":"2. Transformer Overall Architecture","slug":"_2-transformer-overall-architecture","link":"#_2-transformer-overall-architecture","children":[]},{"level":2,"title":"3. Hyperparameters","slug":"_3-hyperparameters","link":"#_3-hyperparameters","children":[]},{"level":2,"title":"4. Tensor Dimensionality Transformation","slug":"_4-tensor-dimensionality-transformation","link":"#_4-tensor-dimensionality-transformation","children":[]},{"level":2,"title":"5. Number of Trainable Parameters","slug":"_5-number-of-trainable-parameters","link":"#_5-number-of-trainable-parameters","children":[{"level":3,"title":"5.1. MultiHeadedAttention","slug":"_5-1-multiheadedattention","link":"#_5-1-multiheadedattention","children":[]},{"level":3,"title":"5.2. PositionwiseFeedForward","slug":"_5-2-positionwisefeedforward","link":"#_5-2-positionwisefeedforward","children":[]},{"level":3,"title":"5.3. LayerNorm","slug":"_5-3-layernorm","link":"#_5-3-layernorm","children":[]},{"level":3,"title":"5.4. Embeddings","slug":"_5-4-embeddings","link":"#_5-4-embeddings","children":[]},{"level":3,"title":"5.5. Total Trainable Parameters","slug":"_5-5-total-trainable-parameters","link":"#_5-5-total-trainable-parameters","children":[]}]},{"level":2,"title":"6. Source Code","slug":"_6-source-code","link":"#_6-source-code","children":[{"level":3,"title":"6.1. Complete Model","slug":"_6-1-complete-model","link":"#_6-1-complete-model","children":[]},{"level":3,"title":"6.2. EncoderDecoder","slug":"_6-2-encoderdecoder","link":"#_6-2-encoderdecoder","children":[]},{"level":3,"title":"6.3. Encoder","slug":"_6-3-encoder","link":"#_6-3-encoder","children":[]},{"level":3,"title":"6.4. Decoder","slug":"_6-4-decoder","link":"#_6-4-decoder","children":[]},{"level":3,"title":"6.5. MultiHeadedAttention","slug":"_6-5-multiheadedattention","link":"#_6-5-multiheadedattention","children":[]},{"level":3,"title":"6.6. PositionwiseFeedForward","slug":"_6-6-positionwisefeedforward","link":"#_6-6-positionwisefeedforward","children":[]},{"level":3,"title":"6.7. Embeddings","slug":"_6-7-embeddings","link":"#_6-7-embeddings","children":[]},{"level":3,"title":"6.8. PositionalEncoding","slug":"_6-8-positionalencoding","link":"#_6-8-positionalencoding","children":[]},{"level":3,"title":"6.9. Generator","slug":"_6-9-generator","link":"#_6-9-generator","children":[]},{"level":3,"title":"6.10. clones","slug":"_6-10-clones","link":"#_6-10-clones","children":[]},{"level":3,"title":"6.11. LayerNorm","slug":"_6-11-layernorm","link":"#_6-11-layernorm","children":[]},{"level":3,"title":"6.12. SublayerConnection","slug":"_6-12-sublayerconnection","link":"#_6-12-sublayerconnection","children":[]},{"level":3,"title":"6.13. Example Usage","slug":"_6-13-example-usage","link":"#_6-13-example-usage","children":[]}]}],"git":{"createdTime":1716567971000,"updatedTime":1717378386000,"contributors":[{"name":"unknown","email":"15721607377@163.com","commits":2}]},"readingTime":{"minutes":16.02,"words":4805},"filePathRelative":"posts/LLM/transformer.md","localizedDate":"May 24, 2024","excerpt":"<h1> Transformer Source Code Exploration</h1>\\n<ul>\\n<li>\\n<ol>\\n<li>About</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"2\\">\\n<li>Transformer Overall Architecture</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"3\\">\\n<li>Hyperparameters</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"4\\">\\n<li>Tensor Dimensionality Transformation</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"5\\">\\n<li>Number of Trainable Parameters</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"6\\">\\n<li>Source Code</li>\\n</ol>\\n</li>\\n</ul>\\n","autoDesc":true}');export{e as data};