import{_ as i,a as l,b as n,c as t,d as s,e as a,f as d}from"./036_mcp_logic-_xrBW_ye.js";import{_ as r}from"./plugin-vue_export-helper-x3n3nnut.js";import{o,c as u,f as c,a as e,b as v,e as m}from"./app-xyQ2iw7y.js";const p={},h=e("h1",{id:"mcp-技术解读",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#mcp-技术解读","aria-hidden":"true"},"#"),v(" MCP 技术解读")],-1),b=e("ul",null,[e("li",null,"MCP概念、演进与意义"),e("li",null,"MCP架构、核心组件与功能类型"),e("li",null,"MCP Client 与 MCP Server"),e("li",null,"不同角色使用 MCP 的方式和逻辑")],-1),g=m('<h2 id="_1-mcp是什么" tabindex="-1"><a class="header-anchor" href="#_1-mcp是什么" aria-hidden="true">#</a> 1. MCP是什么</h2><p>MCP（Model Context Protocol） 是由 Anthropic（就是训练 Claude 的那个公司） 推出的一种新兴的<strong>标准开放协议</strong>，用于解决 Agent 平台的<strong>AI 模型与外部工具交互</strong>的痛点，帮助你在 LLM 之上构建智能体和复杂工作流。MCP 将应用程序向 AI 模型提供上下文的方式标准化，旨在通过安全的双向连接来增强 AI 模型与工具和数据源的交互能力。可以将 MCP 视为 AI 应用的 USB-C 接口，正如 USB-C 为设备连接各种外设和配件提供了标准化方式，MCP 也为 AI 模型连接不同数据源和工具提供了标准化方法。</p><p>MCP Server 就是为了实现 AI Agent 的自动化而存在的，它是一个中间层，告诉 AI Agent 目前存在哪些服务，哪些 API，哪些数据源，AI Agent 可以根据 Server 提供的信息来决定是否调用某个服务，然后通过 Function Calling 来执行函数。以前每个工具都要单独写代码才能连AI，但现在MCP就像&quot;通用接口&quot;，工具和AI各装一个MCP插件，就能直接对话了。</p><p>比如要用Github服务，最终实现还是通过调用Github的API（https://api.github.com）来实现和Github交互，在调用 Github 官方的 API 之前，MCP 的主要工作是描述 Server 提供了哪些能力(给 LLM 提供)，需要哪些参数(参数具体的功能是什么)，最后返回的结果是什么。所以 MCP Server 并不是一个新颖的、高深的东西，它只是一个具有共识的协议。</p><p>官网：https://modelcontextprotocol.io/introduction</p><p>Github: https://github.com/modelcontextprotocol</p><h2 id="_2-ai-工具调用演进" tabindex="-1"><a class="header-anchor" href="#_2-ai-工具调用演进" aria-hidden="true">#</a> 2. AI 工具调用演进</h2><p>从复杂提示词 → Function Calling → GPTs 插件 → MCP 协议的技术演进路径，非常类似于几个经典技术标准化的发展历程，特别是 Web 技术、互联网协议和 API 标准化的过程。以 Web API 的标准化进程类比，</p><table><thead><tr><th>Web API 演进</th><th>AI 工具调用演进</th></tr></thead><tbody><tr><td>早期：RPC（远程过程调用）各自实现</td><td>早期：提示词工程中的工具调用</td></tr><tr><td>发展期：SOAP 和 XML-RPC 等框架</td><td>发展期：Function Calling 的 JSON 结构化调用</td></tr><tr><td>成熟期：REST API 成为主流</td><td>成熟期：GPTs 等平台专属实现</td></tr><tr><td>统一期：GraphQL 等新一代标准</td><td>统一期：MCP 作为统一协议</td></tr></tbody></table><h2 id="_3-mcp-出来前后的区别是啥-意义是啥" tabindex="-1"><a class="header-anchor" href="#_3-mcp-出来前后的区别是啥-意义是啥" aria-hidden="true">#</a> 3. MCP 出来前后的区别是啥？意义是啥？</h2><p>MCP 出来前后的区别</p><ul><li>MCP 出来前，Agent 平台需要自己<strong>维护可用工具列表</strong>以及<strong>进行工具调用</strong></li><li>Agent 平台集成 MCP 后，<strong>维护可用工具列表</strong>以及<strong>进行工具调用</strong>都由MCP来做</li></ul><p>MCP意义</p><ul><li>MCP 解决了 Agent 平台工具调用标准不统一的问题，MCP Server 直接提供功能 API 作为 MCP 工具，供任意 Agent 平台直接使用，避免 Agent 平台重复造轮子，大幅降低开发成本</li><li>MCP 通过标准化协议和客户端-服务器架构，提升了 AI 的上下文感知、自动化能力和安全性</li><li>MCP 的客户端-服务器架构支持模块化扩展，一个客户端可连接多个服务器，Agent 平台可以轻松解锁多样化功能</li></ul><h2 id="_4-mcp-vs-function-calling" tabindex="-1"><a class="header-anchor" href="#_4-mcp-vs-function-calling" aria-hidden="true">#</a> 4. MCP vs Function Calling</h2><p>功能深度不同</p><ul><li>MCP: 连接层</li><li>Function Calling: 执行层</li><li>AI Agent: 决策层</li></ul><p>应用场景不同</p><ul><li>MCP: 需要连接大量工具，但不想写代码</li><li>Function Calling: 简单任务自动化</li><li>AI Agent: 自主决策</li></ul><h2 id="_5-mcp-架构" tabindex="-1"><a class="header-anchor" href="#_5-mcp-架构" aria-hidden="true">#</a> 5. MCP 架构</h2><p>MCP 采用客户端-服务器架构</p><figure><img src="'+i+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><figure><img src="'+l+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><figure><img src="'+n+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h2 id="_6-mcp-核心组件" tabindex="-1"><a class="header-anchor" href="#_6-mcp-核心组件" aria-hidden="true">#</a> 6. MCP 核心组件</h2><p>​核心组件</p><ul><li>Host 主机 <ul><li>是应用程序（如 Claude 桌面或 IDEs），它们发起连接</li></ul></li><li>Client 客户端 <ul><li>在主机应用程序内部与服务器保持 1:1 的连接</li></ul></li><li>Server 服务器 <ul><li>为客户端提供上下文、工具和提示</li></ul></li><li>Protocol layer 协议层 <ul><li>协议层处理消息封装、请求/响应链接和高级通信模式</li></ul></li><li>Transport layer 传输层 <ul><li>传输层负责处理 Client 和 Server 之间的实际通信</li></ul></li></ul><p>MCP 支持多种传输机制，所有传输都使用 JSON-RPC 2.0 来交换消息</p><ul><li>Stdio transport （标准输入输出传输） <ul><li>使用标准输入/输出进行通信</li><li>适用于本地进程</li></ul></li><li>HTTP with SSE transport （使用 SSE 传输的 HTTP） <ul><li>服务器到客户端的消息传输: 服务器发送事件方式（Server-Sent Events，SSE）</li><li>客户端到服务器的消息传输: HTTP POST方式</li></ul></li></ul><p>消息类型</p><ul><li>Requests</li><li>Results</li><li>Errors</li><li>Notifications</li></ul><h2 id="_7-mcp-连接生命周期" tabindex="-1"><a class="header-anchor" href="#_7-mcp-连接生命周期" aria-hidden="true">#</a> 7. MCP 连接生命周期</h2><h3 id="_7-1-initialization-初始化" tabindex="-1"><a class="header-anchor" href="#_7-1-initialization-初始化" aria-hidden="true">#</a> 7.1. Initialization 初始化</h3><figure><img src="'+t+`" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_7-2-message-exchange-消息交换" tabindex="-1"><a class="header-anchor" href="#_7-2-message-exchange-消息交换" aria-hidden="true">#</a> 7.2. Message exchange 消息交换</h3><ul><li>客户端或服务器发送请求，另一方响应</li><li>任一方发送单向消息(Notifications)</li></ul><h3 id="_7-3-termination-终止" tabindex="-1"><a class="header-anchor" href="#_7-3-termination-终止" aria-hidden="true">#</a> 7.3. Termination 终止</h3><p>任何一方均可终止连接</p><h2 id="_8-mcp-功能类型" tabindex="-1"><a class="header-anchor" href="#_8-mcp-功能类型" aria-hidden="true">#</a> 8. MCP 功能类型</h2><p>MCP 服务器提供三种主要功能类型：</p><ul><li>Tools 工具 <ul><li>可以由 LLM 集成的功能</li><li>常见工具类型 <ul><li>System operations 系统操作 <ul><li>与本地系统交互的工具</li></ul></li><li>API integrations API 集成 <ul><li>封装外部 API 的工具</li></ul></li><li>Data processing 数据处理 <ul><li>转换或分析数据的工具</li></ul></li></ul></li><li>https://modelcontextprotocol.io/docs/concepts/tools#python</li></ul></li><li>Resources 资源 <ul><li>可以由 LLM 读取的任何类型的数据</li><li>每个资源都由一个唯一的 URI 标识，可以包含文本或二进制数据</li><li>Resource types 资源类型 <ul><li>Text resources 文本资源 <ul><li>文本资源包含 UTF-8 编码的文本数据。这些适用于： <ul><li>Source code 源代码</li><li>Configuration files 配置文件</li><li>Log files 日志文件</li><li>JSON/XML data JSON/XML 数据</li><li>Plain text 纯文本</li></ul></li></ul></li><li>Binary resources 二进制资源 <ul><li>二进制资源包含原始的二进制数据，以 base64 编码。这些资源适用于： <ul><li>Images 图片</li><li>PDFs PDF 文件</li><li>Audio files 音频文件</li><li>Video files 视频文件</li><li>Other non-text formats 其他非文本格式</li></ul></li></ul></li></ul></li><li>https://modelcontextprotocol.io/docs/concepts/resources</li></ul></li><li>Prompts 提示 <ul><li>创建可重用的提示模板和工作流程，帮助用户完成特定任务的预写提示模板</li><li>https://modelcontextprotocol.io/docs/concepts/prompts</li></ul></li></ul><p>两种额外功能：</p><ul><li>Sampling <ul><li>MCPServer 调用 MCPClient 的 LLM</li><li>https://modelcontextprotocol.io/docs/concepts/sampling</li></ul></li><li>Roots <ul><li>Roots定义了服务器可以操作的范围</li><li>比如客户端在服务器上开了个人工作空间，客户端在工作空间中进行操作，告诉服务器它在用这个工作空间</li><li>Roots通常用于定义： <ul><li>Project directories 项目目录</li><li>Repository locations 仓库位置</li><li>API endpoints API 端点</li><li>Configuration locations 配置位置</li><li>Resource boundaries 资源边界</li></ul></li><li>https://modelcontextprotocol.io/docs/concepts/roots</li></ul></li></ul><h2 id="_9-不同功能类型的标准协议" tabindex="-1"><a class="header-anchor" href="#_9-不同功能类型的标准协议" aria-hidden="true">#</a> 9. 不同功能类型的标准协议</h2><h3 id="_9-1-resources" tabindex="-1"><a class="header-anchor" href="#_9-1-resources" aria-hidden="true">#</a> 9.1. Resources</h3><p>Resources URIs 统一资源标识符</p><ul><li>每个Resource都按以下标准定义</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>[protocol]://[host]/[path]
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_9-2-prompts" tabindex="-1"><a class="header-anchor" href="#_9-2-prompts" aria-hidden="true">#</a> 9.2. Prompts</h3><p>Prompt Structure 提示结构</p><ul><li>每个Prompt都按以下标准定义</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
  name: string;              // Unique identifier for the prompt
  description?: string;      // Human-readable description
  arguments?: [              // Optional list of arguments
    {
      name: string;          // Argument identifier
      description?: string;  // Argument description
      required?: boolean;    // Whether argument is required
    }
  ]
}
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_9-3-tools" tabindex="-1"><a class="header-anchor" href="#_9-3-tools" aria-hidden="true">#</a> 9.3. Tools</h3><p>Tool definition structure 工具定义结构</p><ul><li>每个 Tool 都按以下标准定义</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
  name: string;          // Unique identifier for the tool
  description?: string;  // Human-readable description
  inputSchema: {         // JSON Schema for the tool&#39;s parameters
    type: &quot;object&quot;,
    properties: { ... }  // Tool-specific parameters
  },
  annotations?: {        // Optional hints about tool behavior
    title?: string;      // Human-readable title for the tool
    readOnlyHint?: boolean;    // If true, the tool does not modify its environment
    destructiveHint?: boolean; // If true, the tool may perform destructive updates
    idempotentHint?: boolean;  // If true, repeated calls with same args have no additional effect
    openWorldHint?: boolean;   // If true, tool interacts with external entities
  }
}
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_9-4-sampling" tabindex="-1"><a class="header-anchor" href="#_9-4-sampling" aria-hidden="true">#</a> 9.4. Sampling</h3><p>Message format 消息格式</p><ul><li>服务器发起 Sampling 请求都按以下标准定义</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
  messages: [
    {
      role: &quot;user&quot; | &quot;assistant&quot;,
      content: {
        type: &quot;text&quot; | &quot;image&quot;,

        // For text:
        text?: string,

        // For images:
        data?: string,             // base64 encoded
        mimeType?: string
      }
    }
  ],
  modelPreferences?: {
    hints?: [{
      name?: string                // Suggested model name/family
    }],
    costPriority?: number,         // 0-1, importance of minimizing cost
    speedPriority?: number,        // 0-1, importance of low latency
    intelligencePriority?: number  // 0-1, importance of capabilities
  },
  systemPrompt?: string,
  includeContext?: &quot;none&quot; | &quot;thisServer&quot; | &quot;allServers&quot;,
  temperature?: number,
  maxTokens: number,
  stopSequences?: string[],
  metadata?: Record&lt;string, unknown&gt;
}
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Response format 响应格式</p><ul><li>客户端返回 Sampling 结果都按以下标准定义</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
  model: string,  // Name of the model used
  stopReason?: &quot;endTurn&quot; | &quot;stopSequence&quot; | &quot;maxTokens&quot; | string,
  role: &quot;user&quot; | &quot;assistant&quot;,
  content: {
    type: &quot;text&quot; | &quot;image&quot;,
    text?: string,
    data?: string,
    mimeType?: string
  }
}
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="_10-mcp-client-mcp-客户端" tabindex="-1"><a class="header-anchor" href="#_10-mcp-client-mcp-客户端" aria-hidden="true">#</a> 10. MCP Client (MCP 客户端)</h2><p>比较火的MCP客户端</p><ul><li>适合非工程师 <ul><li>Claude Desktop App <ul><li>Anthropic官方产品，支持MCP的本地工具和数据源集成，适合工程师和非工程师</li><li>https://claude.ai/download</li></ul></li><li>LibreChat <ul><li>开源聊天应用，支持MCP工具集成</li><li>https://github.com/danny-avila/LibreChat</li></ul></li></ul></li><li>适合工程师 <ul><li>Cursor <ul><li>AI驱动的代码编辑器，快速采用MCP，适合开发者</li><li>https://www.cursor.com/</li></ul></li><li>Cline <ul><li>开源VSCode扩展，支持MCP市场和自定义服务器</li><li>https://github.com/cline/cline</li></ul></li><li>Continue <ul><li>VSCode和JetBrains扩展，支持完整的MCP功能</li><li>https://github.com/continuedev/continue</li></ul></li><li>Zed <ul><li>高性能代码编辑器，支持MCP扩展</li><li>https://zed.dev/</li></ul></li><li>Windsurf Editor <ul><li>AI驱动的代码编辑器，支持MCP协议</li><li>https://windsurf.com/editor</li></ul></li><li>Sourcegraph Cody <ul><li>AI编码助手，支持MCP配置</li><li>https://sourcegraph.com/cody</li></ul></li></ul></li></ul><p>应用程序集成对 MCP 支持后（也就是作为 MCP 客户端），可为用户提供强大的 AI 上下文能力，方便用户通过 AI 模型与 MCP 服务器进行交互</p><p>不同客户端可能支持不同的 MCP 功能，因此与 MCP 服务器的集成程度也有所差异。</p><table><thead><tr><th>Client</th><th>Resources</th><th>Prompts</th><th>Tools</th><th>Sampling</th><th>Roots</th><th>Notes</th></tr></thead><tbody><tr><td>Claude Desktop App</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools, prompts, and resources.</td></tr><tr><td>LibreChat</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools for Agents</td></tr><tr><td>Cursor</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools.</td></tr><tr><td>Cline</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools and resources.</td></tr><tr><td>Continue</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools, prompts, and resources.</td></tr><tr><td>Zed</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>Prompts appear as slash commands</td></tr><tr><td>Windsurf Editor</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools with AI Flow for collaborative development.</td></tr><tr><td>Sourcegraph Cody</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>Supports resources through OpenCTX</td></tr></tbody></table><p>More:</p><ul><li>官网提及的客户端列表：https://modelcontextprotocol.io/clients</li><li>PulseMCP提及的客户端列表：https://www.pulsemcp.com/clients</li></ul><h3 id="_10-1-cursor-作为-mcp-客户端" tabindex="-1"><a class="header-anchor" href="#_10-1-cursor-作为-mcp-客户端" aria-hidden="true">#</a> 10.1. Cursor 作为 MCP 客户端</h3><p>Cursor 配置 MCP 的介绍：https://docs.cursor.com/context/model-context-protocol</p><p>将 Cursor 连接到外部工具或数据源，代替了用户通过 Prompt 告诉 Cursor 外部的一些情况</p><p>示例：</p><figure><img src="`+s+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><p>传输机制：</p><figure><img src="'+a+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h2 id="_11-mcp-server-mcp-服务器" tabindex="-1"><a class="header-anchor" href="#_11-mcp-server-mcp-服务器" aria-hidden="true">#</a> 11. MCP Server (MCP 服务器)</h2><ul><li>官方 MCP Servers <ul><li>https://github.com/modelcontextprotocol/servers</li><li>https://modelcontextprotocol.io/examples</li></ul></li><li>Cursor MCP Servers <ul><li>https://cursor.directory/mcp</li></ul></li><li>punkpeye 精选 MCP Servers: https://github.com/punkpeye/awesome-mcp-servers/blob/main/README-zh.md</li><li>Cline的MCP Marketplace</li><li>https://smithery.ai/</li><li>https://www.pulsemcp.com/</li><li>http://glama.ai/mcp/servers</li></ul><p>好用的MCP工具</p><ul><li>数据与文件系统 <ul><li>Filesystem <ul><li>提供安全的文件操作功能，并且可以配置访问控制权限，确保文件访问的安全性和规范性</li></ul></li><li>PostgreSQL <ul><li>提供只读数据库访问，具备架构检查功能</li></ul></li></ul></li><li>开发工具 <ul><li>GitHub <ul><li>实现仓库管理、文件操作，还集成了 GitHub API，方便与 GitHub 平台进行交互</li></ul></li><li>Git <ul><li>提供读取、搜索和操作 Git 仓库的工具，帮助开发人员管理代码版本</li></ul></li></ul></li><li>爬虫 <ul><li>Firecrawl：复制网站</li></ul></li><li>网络与浏览器自动化 <ul><li>Brave Search：利用Brave的搜索API进行网络和本地搜索。</li><li>Fetch：为LLM优化的网络内容获取和转换</li><li>Puppeteer：提供浏览器自动化和网页抓取功能。</li><li>Playwright：控制浏览器 <ul><li>可以自动化浏览器任务。填表单，截图，导航网页。还能执行JavaScript，监控控制台日志</li><li>让LLM和浏览器无缝配合。API测试也变得更简单</li></ul></li></ul></li></ul><h2 id="_12-不同角色使用-mcp-的方式和逻辑" tabindex="-1"><a class="header-anchor" href="#_12-不同角色使用-mcp-的方式和逻辑" aria-hidden="true">#</a> 12. 不同角色使用 MCP 的方式和逻辑</h2><ul><li>MCP Client 用户 （eg. 使用 Claude Desktop App 或 Cursor 的用户） <ul><li>只需在配置中添加要使用的 MCP Server 后，正常问问题就行</li></ul></li><li>MCP Client 开发者 （eg. Claude Desktop App 或 Cursor的开发者） <ul><li>实例化 MCP 客户端</li><li>MCP 客户端与 MCP 服务器建立连接</li><li>每当用户发起一个问题，执行以下逻辑（传给LLM的工具列表是和工具调用是由MCP负责的）： <ul><li>拼接用户 query 为 messages</li><li><strong>MCP Client 获取可用工具列表 available_tools</strong></li><li>将 messages 和 available_tools 作为入参调用 LLM</li><li>LLM 响应有两种：text 或 tool_use</li><li>如果是 text 则将文本结果返回给用户，结束</li><li>如果是 tool_use <ul><li><strong>MCP Client 通过MCP Server执行工具调用</strong></li><li>得到工具调用结果</li><li>messages 除了用户 query（&quot;role&quot;: &quot;user&quot;）外，接着拼接 LLM 响应（&quot;role&quot;: &quot;assistant&quot;）和工具调用结果（&quot;role&quot;: &quot;user&quot;）为新的 messages</li><li>将拼接后的 messages 和 available_tools 作为参数再次调用 LLM</li><li>将 LLM 响应结果返回给用户，结束</li></ul></li></ul></li></ul></li><li>MCP Server 开发者 <ul><li>实例化 MCP 服务器</li><li>定义工具，实现每个工具的执行逻辑</li><li>运行 MCP 服务器</li></ul></li></ul><figure><img src="'+d+`" alt="使用 MCP 的方式和逻辑" tabindex="0" loading="lazy"><figcaption>使用 MCP 的方式和逻辑</figcaption></figure><h2 id="_13-mcp-client-用户使用案例-claude-desktop-app" tabindex="-1"><a class="header-anchor" href="#_13-mcp-client-用户使用案例-claude-desktop-app" aria-hidden="true">#</a> 13. MCP Client 用户使用案例（Claude Desktop App）</h2><h3 id="_13-1-在-claude-desktop-app-中配置想要使用的mcp服务器" tabindex="-1"><a class="header-anchor" href="#_13-1-在-claude-desktop-app-中配置想要使用的mcp服务器" aria-hidden="true">#</a> 13.1. 在 Claude Desktop App 中配置想要使用的MCP服务器</h3><p>在Claude Desktop App 中添加想要使用的 MCP 服务器 &quot;weather&quot; 到 &quot;mcpServers&quot; 配置项</p><ul><li>告诉 Claude Desktop App 有一个名为 &quot;weather&quot; 的 MCP 服务器</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
    &quot;mcpServers&quot;: {
        &quot;weather&quot;: {
            &quot;command&quot;: &quot;uv&quot;,
            &quot;args&quot;: [
                &quot;--directory&quot;,
                &quot;/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather&quot;,
                &quot;run&quot;,
                &quot;weather.py&quot;
            ]
        }
    }
}
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>保存配置后重启Claude Desktop App</p><h3 id="_13-2-在-claude-desktop-app-中查看可用工具" tabindex="-1"><a class="header-anchor" href="#_13-2-在-claude-desktop-app-中查看可用工具" aria-hidden="true">#</a> 13.2. 在 Claude Desktop App 中查看可用工具</h3><p>可以看到配置的所有服务器提供的所有工具</p><ul><li>工具名称</li><li>工具的功能描述</li><li>工具来源于哪个MCP服务器</li></ul><h3 id="_13-3-在-claude-desktop-app-中问一些相关问题" tabindex="-1"><a class="header-anchor" href="#_13-3-在-claude-desktop-app-中问一些相关问题" aria-hidden="true">#</a> 13.3. 在 Claude Desktop App 中问一些相关问题</h3><ul><li>用户发起一个问题</li><li>Claude分析可用工具，决定是否使用工具和使用哪个工具</li><li>如果确定使用MCP工具，该工具对应的MCP Client通过MCP Server执行选中的工具</li><li>执行结果返回给Claude</li><li>Claude将执行结果转为自然语言响应</li><li>用户收到响应</li></ul><h2 id="_14-构建-mcp-服务器案例" tabindex="-1"><a class="header-anchor" href="#_14-构建-mcp-服务器案例" aria-hidden="true">#</a> 14. 构建 MCP 服务器案例</h2><p>以下代码构建的 MCP 服务器名称为 “weather”，提供了2个工具：get-alerts 和 get-forecast</p><h3 id="_14-1-实例化-mcp-服务器" tabindex="-1"><a class="header-anchor" href="#_14-1-实例化-mcp-服务器" aria-hidden="true">#</a> 14.1. 实例化 MCP 服务器</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>mcp = FastMCP(&quot;weather&quot;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_14-2-定义工具-实现每个工具的执行逻辑" tabindex="-1"><a class="header-anchor" href="#_14-2-定义工具-实现每个工具的执行逻辑" aria-hidden="true">#</a> 14.2. 定义工具，实现每个工具的执行逻辑</h3><ul><li>加上@mcp.tool()后，一个API就是一个MCP工具</li><li>不需要提供一个查询可用工具列表的API</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>@mcp.tool()
async def get_alerts(state: str) -&gt; str:
    &quot;&quot;&quot;Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    &quot;&quot;&quot;
    pass

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -&gt; str:
    &quot;&quot;&quot;Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    &quot;&quot;&quot;
    pass
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_14-3-运行-mcp-服务器" tabindex="-1"><a class="header-anchor" href="#_14-3-运行-mcp-服务器" aria-hidden="true">#</a> 14.3. 运行 MCP 服务器</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>mcp.run(transport=&#39;stdio&#39;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h2 id="_15-构建-mcp-客户端案例" tabindex="-1"><a class="header-anchor" href="#_15-构建-mcp-客户端案例" aria-hidden="true">#</a> 15. 构建 MCP 客户端案例</h2><h3 id="_15-1-创建-env-文件" tabindex="-1"><a class="header-anchor" href="#_15-1-创建-env-文件" aria-hidden="true">#</a> 15.1. 创建 .env 文件</h3><p>设置Anthropic API key</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>ANTHROPIC_API_KEY=&lt;your key here&gt;
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_15-2-定义-mcpclient-类" tabindex="-1"><a class="header-anchor" href="#_15-2-定义-mcpclient-类" aria-hidden="true">#</a> 15.2. 定义 MCPClient 类</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-3-实现服务器连接管理函数-connect-to-server" tabindex="-1"><a class="header-anchor" href="#_15-3-实现服务器连接管理函数-connect-to-server" aria-hidden="true">#</a> 15.3. 实现服务器连接管理函数 connect_to_server</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def connect_to_server(self, server_script_path: str):
    &quot;&quot;&quot;Connect to an MCP server

    Args:
        server_script_path: Path to the server script (.py or .js)
    &quot;&quot;&quot;
    is_python = server_script_path.endswith(&#39;.py&#39;)
    is_js = server_script_path.endswith(&#39;.js&#39;)
    if not (is_python or is_js):
        raise ValueError(&quot;Server script must be a .py or .js file&quot;)

    command = &quot;python&quot; if is_python else &quot;node&quot;
    server_params = StdioServerParameters(
        command=command,
        args=[server_script_path],
        env=None
    )

    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
    self.stdio, self.write = stdio_transport
    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

    await self.session.initialize()

    # List available tools
    response = await self.session.list_tools()
    tools = response.tools
    print(&quot;\\nConnected to server with tools:&quot;, [tool.name for tool in tools])
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-4-实现处理查询和工具调用的函数-process-query" tabindex="-1"><a class="header-anchor" href="#_15-4-实现处理查询和工具调用的函数-process-query" aria-hidden="true">#</a> 15.4. 实现处理查询和工具调用的函数 process_query</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def process_query(self, query: str) -&gt; str:
    &quot;&quot;&quot;Process a query using Claude and available tools&quot;&quot;&quot;
    messages = [
        {
            &quot;role&quot;: &quot;user&quot;,
            &quot;content&quot;: query
        }
    ]

    response = await self.session.list_tools()
    available_tools = [{
        &quot;name&quot;: tool.name,
        &quot;description&quot;: tool.description,
        &quot;input_schema&quot;: tool.inputSchema
    } for tool in response.tools]

    # Initial Claude API call
    response = self.anthropic.messages.create(
        model=&quot;claude-3-5-sonnet-20241022&quot;,
        max_tokens=1000,
        messages=messages,
        tools=available_tools
    )

    # Process response and handle tool calls
    final_text = []

    assistant_message_content = []
    for content in response.content:
        if content.type == &#39;text&#39;:
            final_text.append(content.text)
            assistant_message_content.append(content)
        elif content.type == &#39;tool_use&#39;:
            tool_name = content.name
            tool_args = content.input

            # Execute tool call
            result = await self.session.call_tool(tool_name, tool_args)
            final_text.append(f&quot;[Calling tool {tool_name} with args {tool_args}]&quot;)

            assistant_message_content.append(content)
            messages.append({
                &quot;role&quot;: &quot;assistant&quot;,
                &quot;content&quot;: assistant_message_content
            })
            messages.append({
                &quot;role&quot;: &quot;user&quot;,
                &quot;content&quot;: [
                    {
                        &quot;type&quot;: &quot;tool_result&quot;,
                        &quot;tool_use_id&quot;: content.id,
                        &quot;content&quot;: result.content
                    }
                ]
            })

            # Get next response from Claude
            response = self.anthropic.messages.create(
                model=&quot;claude-3-5-sonnet-20241022&quot;,
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

            final_text.append(response.content[0].text)

    return &quot;\\n&quot;.join(final_text)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-5-定义-chat-loop-和-cleanup-函数" tabindex="-1"><a class="header-anchor" href="#_15-5-定义-chat-loop-和-cleanup-函数" aria-hidden="true">#</a> 15.5. 定义 chat_loop 和 cleanup 函数</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def chat_loop(self):
    &quot;&quot;&quot;Run an interactive chat loop&quot;&quot;&quot;
    print(&quot;\\nMCP Client Started!&quot;)
    print(&quot;Type your queries or &#39;quit&#39; to exit.&quot;)

    while True:
        try:
            query = input(&quot;\\nQuery: &quot;).strip()

            if query.lower() == &#39;quit&#39;:
                break

            response = await self.process_query(query)
            print(&quot;\\n&quot; + response)

        except Exception as e:
            print(f&quot;\\nError: {str(e)}&quot;)

async def cleanup(self):
    &quot;&quot;&quot;Clean up resources&quot;&quot;&quot;
    await self.exit_stack.aclose()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-6-主入口" tabindex="-1"><a class="header-anchor" href="#_15-6-主入口" aria-hidden="true">#</a> 15.6. 主入口</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def main():
    if len(sys.argv) &lt; 2:
        print(&quot;Usage: python client.py &lt;path_to_server_script&gt;&quot;)
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == &quot;__main__&quot;:
    import sys
    asyncio.run(main())
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,119);function _(q,C){return o(),u("div",null,[h,b,c(" more "),g])}const M=r(p,[["render",_],["__file","036_mcp.html.vue"]]);export{M as default};
