const t=JSON.parse('{"key":"v-1aae202e","path":"/posts/pytorch/03_pytorch_operation.html","title":"Tensor Operations","lang":"en-US","frontmatter":{"icon":"lightbulb","date":"2022-07-18T00:00:00.000Z","category":["Pytorch"],"tag":["Pytorch"],"description":"Tensor Operations Scalar, Vector, Matrix, and Tensor Initializing Tensors Attributes of Tensors Basic Operations on Tensors Summation and Averaging Product Operations Calculating the Norm of a Vector Gradient Computation","head":[["link",{"rel":"alternate","hreflang":"zh-cn","href":"https://liz-starfield.github.io/blog/zh/posts/pytorch/03_pytorch_operation.html"}],["meta",{"property":"og:url","content":"https://liz-starfield.github.io/blog/posts/pytorch/03_pytorch_operation.html"}],["meta",{"property":"og:site_name","content":"Liz"}],["meta",{"property":"og:title","content":"Tensor Operations"}],["meta",{"property":"og:description","content":"Tensor Operations Scalar, Vector, Matrix, and Tensor Initializing Tensors Attributes of Tensors Basic Operations on Tensors Summation and Averaging Product Operations Calculating the Norm of a Vector Gradient Computation"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"og:locale:alternate","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2024-02-29T04:35:55.000Z"}],["meta",{"property":"article:author","content":"Liz"}],["meta",{"property":"article:tag","content":"Pytorch"}],["meta",{"property":"article:published_time","content":"2022-07-18T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2024-02-29T04:35:55.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"Tensor Operations\\",\\"image\\":[\\"\\"],\\"datePublished\\":\\"2022-07-18T00:00:00.000Z\\",\\"dateModified\\":\\"2024-02-29T04:35:55.000Z\\",\\"author\\":[{\\"@type\\":\\"Person\\",\\"name\\":\\"Liz\\",\\"url\\":\\"https://github.com/liz-starfield\\"}]}"]]},"headers":[{"level":2,"title":"1. Scalar, Vector, Matrix, and Tensor","slug":"_1-scalar-vector-matrix-and-tensor","link":"#_1-scalar-vector-matrix-and-tensor","children":[{"level":3,"title":"1.1. 张量（tensor，n维数组）","slug":"_1-1-张量-tensor-n维数组","link":"#_1-1-张量-tensor-n维数组","children":[]},{"level":3,"title":"1.2. 标量","slug":"_1-2-标量","link":"#_1-2-标量","children":[]},{"level":3,"title":"1.3. 向量","slug":"_1-3-向量","link":"#_1-3-向量","children":[]},{"level":3,"title":"1.4. 矩阵","slug":"_1-4-矩阵","link":"#_1-4-矩阵","children":[]}]},{"level":2,"title":"2. Initializing Tensors","slug":"_2-initializing-tensors","link":"#_2-initializing-tensors","children":[]},{"level":2,"title":"3. Attributes of Tensors","slug":"_3-attributes-of-tensors","link":"#_3-attributes-of-tensors","children":[]},{"level":2,"title":"4. Basic Operations on Tensors","slug":"_4-basic-operations-on-tensors","link":"#_4-basic-operations-on-tensors","children":[]},{"level":2,"title":"5. Summation and Averaging","slug":"_5-summation-and-averaging","link":"#_5-summation-and-averaging","children":[{"level":3,"title":"5.1. 降维求和与降维求平均值","slug":"_5-1-降维求和与降维求平均值","link":"#_5-1-降维求和与降维求平均值","children":[]},{"level":3,"title":"5.2. 非降维求和（累积求和）","slug":"_5-2-非降维求和-累积求和","link":"#_5-2-非降维求和-累积求和","children":[]}]},{"level":2,"title":"6. Product Operations","slug":"_6-product-operations","link":"#_6-product-operations","children":[{"level":3,"title":"6.1. 向量*向量（点积，dot product）","slug":"_6-1-向量-向量-点积-dot-product","link":"#_6-1-向量-向量-点积-dot-product","children":[]},{"level":3,"title":"6.2. 矩阵*向量（matrix-vector product）","slug":"_6-2-矩阵-向量-matrix-vector-product","link":"#_6-2-矩阵-向量-matrix-vector-product","children":[]},{"level":3,"title":"6.3. 矩阵*矩阵（matrix-matrix multiplication）","slug":"_6-3-矩阵-矩阵-matrix-matrix-multiplication","link":"#_6-3-矩阵-矩阵-matrix-matrix-multiplication","children":[]},{"level":3,"title":"6.4. 张量*张量","slug":"_6-4-张量-张量","link":"#_6-4-张量-张量","children":[]}]},{"level":2,"title":"7. Calculating the Norm of a Vector","slug":"_7-calculating-the-norm-of-a-vector","link":"#_7-calculating-the-norm-of-a-vector","children":[{"level":3,"title":"7.1. 向量每个元素的平方和的平方根（又称，L2范数）","slug":"_7-1-向量每个元素的平方和的平方根-又称-l2范数","link":"#_7-1-向量每个元素的平方和的平方根-又称-l2范数","children":[]},{"level":3,"title":"7.2. 向量每个元素的绝对值之和（又称，L1范数）","slug":"_7-2-向量每个元素的绝对值之和-又称-l1范数","link":"#_7-2-向量每个元素的绝对值之和-又称-l1范数","children":[]}]},{"level":2,"title":"8. Gradient Computation","slug":"_8-gradient-computation","link":"#_8-gradient-computation","children":[{"level":3,"title":"8.1. 微积分","slug":"_8-1-微积分","link":"#_8-1-微积分","children":[]},{"level":3,"title":"8.2. 自动微分","slug":"_8-2-自动微分","link":"#_8-2-自动微分","children":[]}]}],"git":{"createdTime":1709181355000,"updatedTime":1709181355000,"contributors":[{"name":"unknown","email":"15721607377@163.com","commits":1}]},"readingTime":{"minutes":9.52,"words":2855},"filePathRelative":"posts/pytorch/03_pytorch_operation.md","localizedDate":"July 18, 2022","excerpt":"<h1> Tensor Operations</h1>\\n<ul>\\n<li>\\n<ol>\\n<li>Scalar, Vector, Matrix, and Tensor</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"2\\">\\n<li>Initializing Tensors</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"3\\">\\n<li>Attributes of Tensors</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"4\\">\\n<li>Basic Operations on Tensors</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"5\\">\\n<li>Summation and Averaging</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"6\\">\\n<li>Product Operations</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"7\\">\\n<li>Calculating the Norm of a Vector</li>\\n</ol>\\n</li>\\n<li>\\n<ol start=\\"8\\">\\n<li>Gradient Computation</li>\\n</ol>\\n</li>\\n</ul>\\n","autoDesc":true}');export{t as data};