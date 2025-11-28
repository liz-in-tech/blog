import{_ as a}from"./plugin-vue_export-helper-x3n3nnut.js";import{o as l,c as i,f as n,a as e,b as s,e as d}from"./app-hKujL8Ix.js";const r={},t=e("h1",{id:"ollama操作指南",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#ollama操作指南","aria-hidden":"true"},"#"),s(" Ollama操作指南")],-1),o=e("ul",null,[e("li",null,"是什么"),e("li",null,"作用与特点"),e("li",null,"怎么使用")],-1),m=d(`<h2 id="_1-about" tabindex="-1"><a class="header-anchor" href="#_1-about" aria-hidden="true">#</a> 1. About</h2><p>官网：https://ollama.com/</p><p>Github：https://github.com/ollama/ollama</p><p>模型库：https://ollama.com/search</p><p>模型库显示信息</p><ul><li>模型分类：All/Embedding/Vision/Tools</li><li>模型排序：Popular/Newest</li><li>模型名称</li><li>模型描述</li><li>可选的参数量规模</li><li>不同规模和变体版本的模型大小、唯一标识、上传时间和下载命令</li><li>被下载次数（表示了受欢迎程度）</li><li>Tags数（可下载的不同规模和变体版本的模型个数）</li><li>更新日期</li></ul><h2 id="_2-what" tabindex="-1"><a class="header-anchor" href="#_2-what" aria-hidden="true">#</a> 2. What</h2><p>用于本地部署开源大模型</p><p>开发者可以在本地测试和调试模型</p><p>用户可以在不联网的情况下使用语言模型，确保数据隐私，响应速度快，完全免费</p><p>特点</p><ul><li>操作简单，简化了模型部署和管理，使用门槛低</li><li>支持多平台：macOS, Linux, and Windows</li><li>模型库很全面且更新很快，除了模型库已有的模型，用户可以在HuggingFace下载gguf模型，可以部署本地已下载好的gguf模型文件，也可以创建自定义模型</li></ul><h2 id="_3-how-to-use-macos平台操作" tabindex="-1"><a class="header-anchor" href="#_3-how-to-use-macos平台操作" aria-hidden="true">#</a> 3. How To Use（macOS平台操作）</h2><h3 id="_3-1-安装ollama" tabindex="-1"><a class="header-anchor" href="#_3-1-安装ollama" aria-hidden="true">#</a> 3.1. 安装Ollama</h3><p>1.在官网下载，得到Ollama-darwin.zip，双击得到Ollama.app，移入Applications<br> 2.第一次打开会提示安装命令行工具，输入密码授权，安装完成</p><h3 id="_3-2-下载模型" tabindex="-1"><a class="header-anchor" href="#_3-2-下载模型" aria-hidden="true">#</a> 3.2. 下载模型</h3><h4 id="_3-2-1-从模型库下载" tabindex="-1"><a class="header-anchor" href="#_3-2-1-从模型库下载" aria-hidden="true">#</a> 3.2.1. 从模型库下载</h4><p>1.从模型库找到想要下载到本地的模型（下载到了用户目录下的.ollama隐藏文件夹下）<br> 2.复制下载命令到命令行后开始下载（一般为run或pull命令）</p><h4 id="_3-2-2-从huggingface下载" tabindex="-1"><a class="header-anchor" href="#_3-2-2-从huggingface下载" aria-hidden="true">#</a> 3.2.2. 从HuggingFace下载</h4><p>1.从HuggingFace（https://huggingface.co/models）中搜索GGUF模型<br> 2.在命令行执行命令</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>ollama run hf.co/{username}/{repository}
ollama run hf.co/{username}/{repository}:{quantization} 
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_3-3-创建自定义模型并push到模型库" tabindex="-1"><a class="header-anchor" href="#_3-3-创建自定义模型并push到模型库" aria-hidden="true">#</a> 3.3. 创建自定义模型并push到模型库</h3><p>1.创建模型文件Modelfile</p><ul><li>FROM：基础模型 <ul><li>FROM ./vicuna-33b.Q4_0.gguf # Ollama 支持在 Modelfile 中导入 GGUF 模型，指向要导入的GGUF模型的本地文件路径</li><li>FROM llama3.2</li></ul></li><li>PARAMETER：参数配置</li><li>SYSTEM：系统指令 system prompt</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM &quot;&quot;&quot;
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
&quot;&quot;&quot;
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>2.执行创建自定义模型命令</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>ollama create my_model_name -f ./Modelfile
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><p>3.运行自定义模型</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>ollama run my_model_name
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><p>4.上传到官网模型库</p><p>1）注册并登录官网，此处的名称即为上传模型的命名空间my_namespace<br> 2）进入My models界面New一个模型, 模型名和刚才自定义的my_model_name一致，按提示在官网Settings设置公钥<br> 3）把模型复制到命名空间下</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>ollama cp my_model_name my_namespace/my_model_name
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><p>4）push模型到模型库</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>ollama push my_namespace/my_model_name
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><p>5）上传成功，如果选择公开模型的话，在模型库中可以搜到自己的模型了</p><h3 id="_3-4-管理并使用模型" tabindex="-1"><a class="header-anchor" href="#_3-4-管理并使用模型" aria-hidden="true">#</a> 3.4. 管理并使用模型</h3><ul><li>增 <ul><li>pull</li><li>run</li><li>cp</li><li>create</li></ul></li><li>删 <ul><li>rm</li></ul></li><li>查 <ul><li>查列表：list</li><li>查运行中的模型列表：ps</li><li>查详情：show</li></ul></li><li>使用 <ul><li>run</li><li>退出聊天：Use Ctrl + d or /bye to exit</li><li>stop</li></ul></li></ul><h3 id="_3-5-更多命令" tabindex="-1"><a class="header-anchor" href="#_3-5-更多命令" aria-hidden="true">#</a> 3.5. 更多命令</h3><p>命令窗口输入 ollama 即可看到所有用法</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama （和运行桌面应用是一样的）
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use &quot;ollama [command] --help&quot; for more information about a command.
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,40);function u(c,h){return l(),i("div",null,[t,o,n(" more "),m])}const b=a(r,[["render",u],["__file","019_ollama.html.vue"]]);export{b as default};
