import{_ as i,a as n,b as t,c as s,d as l,e as a,f as o}from"./036_mcp_logic-_xrBW_ye.js";import{_ as r}from"./plugin-vue_export-helper-x3n3nnut.js";import{o as d,c,f as u,a as e,b as v,e as m}from"./app-xaZLmYxW.js";const p={},h=e("h1",{id:"mcp-technical-overview",tabindex:"-1"},[e("a",{class:"header-anchor",href:"#mcp-technical-overview","aria-hidden":"true"},"#"),v(" MCP Technical Overview")],-1),b=e("ul",null,[e("li",null,"Concept, Evolution, and Significance of MCP"),e("li",null,"MCP Architecture, Core Components, and Function Types"),e("li",null,"MCP Client and MCP Server"),e("li",null,"How Different Roles Use MCP")],-1),g=m('<h2 id="_1-what-is-mcp" tabindex="-1"><a class="header-anchor" href="#_1-what-is-mcp" aria-hidden="true">#</a> 1. What is MCP?</h2><p>MCP (Model Context Protocol) is an emerging <strong>standard open protocol</strong> launched by Anthropic (the company behind Claude) to address the pain points in <strong>AI model and external tool interactions</strong> on Agent platforms. It helps you build intelligent agents and complex workflows on top of LLMs. MCP standardizes how applications provide context to AI models and aims to enhance model interaction with tools and data sources through secure bi-directional connections.</p><p>Think of MCP as the USB-C for AI applications. Just as USB-C standardizes how devices connect to peripherals, MCP standardizes how AI models connect to diverse tools and data sources.</p><p>The MCP Server exists to automate AI Agents. It is a middleware layer that informs the AI Agent about available services, APIs, and data sources. The Agent then decides which to invoke and performs function calls accordingly. In the past, every tool needed custom code to connect with AI. With MCP, tools and AIs can &quot;talk&quot; directly using a shared plugin—like a universal interface.</p><p>For example, to use GitHub, you ultimately call its official API (https://api.github.com). Before calling it, MCP describes to the AI what capabilities the server provides, what parameters are needed, and what results can be expected. MCP Server itself is not a magical invention—it’s simply a consensus-driven protocol.</p><p>Website: https://modelcontextprotocol.io/introduction</p><p>GitHub: https://github.com/modelcontextprotocol</p><h2 id="_2-ai-tool-invocation-evolution" tabindex="-1"><a class="header-anchor" href="#_2-ai-tool-invocation-evolution" aria-hidden="true">#</a> 2. AI Tool Invocation Evolution</h2><p>The technical evolution path from complex prompts → Function Calling → GPTs plugins → MCP protocol is quite similar to the standardization process of several classic technologies — especially Web technologies, Internet protocols, and API standardization. This process can be compared to the standardization journey of Web APIs:</p><table><thead><tr><th>Web API Evolution</th><th>AI Tool Invocation Evolution</th></tr></thead><tbody><tr><td>Early Stage: RPC (Remote Procedure Call), various implementations</td><td>Early Stage: Tool invocation via prompt engineering</td></tr><tr><td>Development Stage: Frameworks like SOAP and XML-RPC</td><td>Development Stage: Structured Function Calling using JSON</td></tr><tr><td>Maturity Stage: REST API becomes mainstream</td><td>Maturity Stage: Platform-specific implementations like GPTs</td></tr><tr><td>Unification Stage: New standards like GraphQL</td><td>Unification Stage: MCP as a unified protocol</td></tr></tbody></table><h2 id="_3-what-s-the-difference-before-and-after-mcp-why-does-it-matter" tabindex="-1"><a class="header-anchor" href="#_3-what-s-the-difference-before-and-after-mcp-why-does-it-matter" aria-hidden="true">#</a> 3. What&#39;s the Difference Before and After MCP? Why Does It Matter?</h2><ul><li>Before MCP: Agent platforms had to <strong>maintain a list of available tools</strong> themselves and <strong>handle tool invocation logic</strong> manually</li><li>After MCP: Both <strong>tool list management</strong> and <strong>invocation logic</strong> are handled by MCP.</li></ul><p>Significance:</p><ul><li>MCP solves the problem of inconsistent tool invocation standards across Agent platforms. MCP Servers expose APIs as MCP tools that any Agent platform can use—eliminating redundant work and significantly lowering development costs.</li><li>MCP improves context-awareness, automation, and security through a standardized client-server protocol.</li><li>Its modular design allows one client to connect to multiple servers, unlocking diverse functionality easily.</li></ul><h2 id="_4-mcp-vs-function-calling" tabindex="-1"><a class="header-anchor" href="#_4-mcp-vs-function-calling" aria-hidden="true">#</a> 4. MCP vs Function Calling</h2><p>Functionality Layers:</p><ul><li><strong>MCP:</strong> Connection Layer</li><li><strong>Function Calling:</strong> Execution Layer</li><li><strong>AI Agent:</strong> Decision Layer</li></ul><p>Use Case Differences:</p><ul><li><strong>MCP:</strong> When connecting many tools without writing a lot of custom code</li><li><strong>Function Calling:</strong> Automating simple tasks</li><li><strong>AI Agent:</strong> Making autonomous decisions</li></ul><h2 id="_5-mcp-architecture" tabindex="-1"><a class="header-anchor" href="#_5-mcp-architecture" aria-hidden="true">#</a> 5. MCP Architecture</h2><p>MCP adopts a client-server architecture.</p><p><img src="'+i+'" alt="" loading="lazy"><br><img src="'+n+'" alt="" loading="lazy"><br><img src="'+t+'" alt="" loading="lazy"></p><h2 id="_6-core-components-of-mcp" tabindex="-1"><a class="header-anchor" href="#_6-core-components-of-mcp" aria-hidden="true">#</a> 6. Core Components of MCP</h2><p>Core Components:</p><ul><li><strong>Host:</strong><br> The application (e.g., Claude Desktop or IDEs) that initiates the connection.</li><li><strong>Client:</strong><br> Resides inside the host application and connects 1:1 with a server.</li><li><strong>Server:</strong><br> Provides context, tools, and prompts to the client.</li><li><strong>Protocol Layer:</strong><br> Handles message packaging, request/response linking, and high-level communication patterns.</li><li><strong>Transport Layer:</strong><br> Manages actual communication between Client and Server.</li></ul><p>MCP supports multiple transport mechanisms. All use JSON-RPC 2.0 for message exchange:</p><p>Transport Types:</p><ul><li><strong>Stdio Transport:</strong><br> Uses standard input/output; suited for local processes.</li><li><strong>HTTP with SSE Transport:</strong><ul><li>Server → Client: via Server-Sent Events (SSE)</li><li>Client → Server: via HTTP POST</li></ul></li></ul><p>Message Types:</p><ul><li>Requests</li><li>Results</li><li>Errors</li><li>Notifications</li></ul><h2 id="_7-mcp-connection-lifecycle" tabindex="-1"><a class="header-anchor" href="#_7-mcp-connection-lifecycle" aria-hidden="true">#</a> 7. MCP Connection Lifecycle</h2><h3 id="_7-1-initialization" tabindex="-1"><a class="header-anchor" href="#_7-1-initialization" aria-hidden="true">#</a> 7.1 Initialization</h3><figure><img src="'+s+`" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h3 id="_7-2-message-exchange" tabindex="-1"><a class="header-anchor" href="#_7-2-message-exchange" aria-hidden="true">#</a> 7.2 Message Exchange</h3><ul><li>Either side can send requests, with the other responding.</li><li>Either side can send one-way notifications.</li></ul><h3 id="_7-3-termination" tabindex="-1"><a class="header-anchor" href="#_7-3-termination" aria-hidden="true">#</a> 7.3 Termination</h3><p>Either party can terminate the connection.</p><h2 id="_8-mcp-function-types" tabindex="-1"><a class="header-anchor" href="#_8-mcp-function-types" aria-hidden="true">#</a> 8. MCP Function Types</h2><p>MCP Servers expose three main function types:</p><ul><li>Tools <ul><li>Executable by LLMs</li><li>Common Tool Types:</li><li><strong>System Operations:</strong> Interact with local systems</li><li><strong>API Integrations:</strong> Wrap external APIs</li><li><strong>Data Processing:</strong> Transform or analyze data</li><li>Docs: https://modelcontextprotocol.io/docs/concepts/tools#python</li></ul></li><li>Resources <ul><li>Any data type readable by LLMs</li><li>Identified via unique URIs; may contain text or binary data</li><li>Resource Types:</li><li><strong>Text Resources:</strong><br> UTF-8 encoded text. Suitable for: <ul><li>Source code</li><li>Config files</li><li>Logs</li><li>JSON/XML</li><li>Plain text</li></ul></li><li><strong>Binary Resources:</strong><br> Base64-encoded binary data. Suitable for: <ul><li>Images</li><li>PDFs</li><li>Audio/Video files</li><li>Other non-text formats</li></ul></li><li>Docs: https://modelcontextprotocol.io/docs/concepts/resources</li></ul></li><li>Prompts <ul><li>Pre-written reusable templates and workflows to assist in specific tasks</li><li>Docs: https://modelcontextprotocol.io/docs/concepts/prompts</li></ul></li></ul><p>Additional Functions</p><ul><li>Sampling <ul><li>Server invokes the LLM hosted on the client side</li><li>https://modelcontextprotocol.io/docs/concepts/sampling</li></ul></li><li>Roots <ul><li>Define the operational scope for the server</li><li>e.g., a workspace that the client uses and informs the server about</li><li>Common Uses: <ul><li>Project directories</li><li>Repo locations</li><li>API endpoints</li><li>Config locations</li><li>Resource boundaries</li></ul></li><li>https://modelcontextprotocol.io/docs/concepts/roots</li></ul></li></ul><h2 id="_9-standard-protocols-for-different-mcp-function-types" tabindex="-1"><a class="header-anchor" href="#_9-standard-protocols-for-different-mcp-function-types" aria-hidden="true">#</a> 9. Standard Protocols for Different MCP Function Types</h2><h3 id="_9-1-resources" tabindex="-1"><a class="header-anchor" href="#_9-1-resources" aria-hidden="true">#</a> 9.1. Resources</h3><p>Resources URIs</p><ul><li>Each Resource is defined according to the following standard</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>[protocol]://[host]/[path]
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_9-2-prompts" tabindex="-1"><a class="header-anchor" href="#_9-2-prompts" aria-hidden="true">#</a> 9.2. Prompts</h3><p>Prompt Structure</p><ul><li>Each Prompt is defined according to the following standard</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_9-3-tools" tabindex="-1"><a class="header-anchor" href="#_9-3-tools" aria-hidden="true">#</a> 9.3. Tools</h3><p>Tool definition structure</p><ul><li>Each Tool is defined according to the following standard</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_9-4-sampling" tabindex="-1"><a class="header-anchor" href="#_9-4-sampling" aria-hidden="true">#</a> 9.4. Sampling</h3><p>Message format</p><ul><li>Each Sampling request initiated by the server follows the same standard.</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Response format</p><ul><li>Each Sampling result returned by the client also follows the same standard.</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="_10-mcp-client" tabindex="-1"><a class="header-anchor" href="#_10-mcp-client" aria-hidden="true">#</a> 10. MCP Client</h2><p>Popular MCP Clients</p><ul><li>For Non-Engineers <ul><li>Claude Desktop App <ul><li>Official product by Anthropic that integrates local tools and data sources using MCP. Suitable for both technical and non-technical users.</li><li>https://claude.ai/download</li></ul></li><li>LibreChat <ul><li>Open-source chat application with MCP tool integration support</li><li>https://github.com/danny-avila/LibreChat</li></ul></li></ul></li><li>For Engineers <ul><li>Cursor <ul><li>AI-powered code editor, early adopter of MCP, ideal for developers</li><li>https://www.cursor.com/</li></ul></li><li>Cline <ul><li>Open-source VSCode extension that supports the MCP marketplace and custom servers</li><li>https://github.com/cline/cline</li></ul></li><li>Continue <ul><li>VSCode and JetBrains extension that supports full MCP functionality</li><li>https://github.com/continuedev/continue</li></ul></li><li>Zed <ul><li>High-performance code editor with MCP extension support</li><li>https://zed.dev/</li></ul></li><li>Windsurf Editor <ul><li>AI-driven code editor that supports the MCP protocol</li><li>https://windsurf.com/editor</li></ul></li><li>Sourcegraph Cody <ul><li>AI coding assistant with MCP configuration support</li><li>https://sourcegraph.com/cody</li></ul></li></ul></li></ul><p>Once an application integrates MCP support (i.e., becomes an MCP Client), it can provide users with powerful AI context capabilities, enabling seamless interaction between the AI model and the MCP Server.</p><p>Different clients may support different MCP functions, so the level of integration with MCP servers may vary.</p><table><thead><tr><th>Client</th><th>Resources</th><th>Prompts</th><th>Tools</th><th>Sampling</th><th>Roots</th><th>Notes</th></tr></thead><tbody><tr><td>Claude Desktop App</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools, prompts, and resources.</td></tr><tr><td>LibreChat</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools for Agents</td></tr><tr><td>Cursor</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools.</td></tr><tr><td>Cline</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools and resources.</td></tr><tr><td>Continue</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools, prompts, and resources.</td></tr><tr><td>Zed</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>Prompts appear as slash commands</td></tr><tr><td>Windsurf Editor</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>Supports tools with AI Flow for collaborative development.</td></tr><tr><td>Sourcegraph Cody</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>Supports resources through OpenCTX</td></tr></tbody></table><p>More:</p><ul><li>Official client list：https://modelcontextprotocol.io/clients</li><li>PulseMCP client list：https://www.pulsemcp.com/clients</li></ul><h3 id="_10-1-cursor-as-an-mcp-client" tabindex="-1"><a class="header-anchor" href="#_10-1-cursor-as-an-mcp-client" aria-hidden="true">#</a> 10.1 Cursor as an MCP Client</h3><p>Cursor’s guide to MCP configuration: https://docs.cursor.com/context/model-context-protocol</p><p>With MCP, Cursor can connect to external tools or data sources, eliminating the need for users to explain external context via prompts.</p><p>Example:</p><figure><img src="`+l+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><p>Transport Mechanism:</p><figure><img src="'+a+'" alt="" tabindex="0" loading="lazy"><figcaption></figcaption></figure><h2 id="_11-mcp-server" tabindex="-1"><a class="header-anchor" href="#_11-mcp-server" aria-hidden="true">#</a> 11. MCP Server</h2><ul><li>Official MCP Servers <ul><li>https://github.com/modelcontextprotocol/servers</li><li>https://modelcontextprotocol.io/examples</li></ul></li><li>Cursor MCP Servers <ul><li>https://cursor.directory/mcp</li></ul></li><li>Curated by punkpeye: https://github.com/punkpeye/awesome-mcp-servers/blob/main/README-zh.md</li><li>Cline’s MCP Marketplace</li><li>https://smithery.ai/</li><li>https://www.pulsemcp.com/</li><li>http://glama.ai/mcp/servers</li></ul><p>Useful MCP Tools</p><ul><li><strong>Data and File Systems</strong><ul><li><strong>Filesystem</strong><ul><li>Provides secure file operations with configurable access controls to ensure safe and compliant access.</li></ul></li><li><strong>PostgreSQL</strong><ul><li>Offers read-only database access with schema introspection.</li></ul></li></ul></li><li><strong>Development Tools</strong><ul><li><strong>GitHub</strong><ul><li>Enables repository management and file operations with integrated GitHub API support.</li></ul></li><li><strong>Git</strong><ul><li>Tools for reading, searching, and manipulating Git repositories, helping developers manage version control.</li></ul></li></ul></li><li><strong>Web Crawlers</strong><ul><li><strong>Firecrawl</strong>: Clone websites.</li></ul></li><li><strong>Networking and Browser Automation</strong><ul><li><strong>Brave Search</strong>: Uses Brave&#39;s API for web and local search.</li><li><strong>Fetch</strong>: Web content retrieval and transformation optimized for LLMs.</li><li><strong>Puppeteer</strong>: Provides browser automation and web scraping.</li><li><strong>Playwright</strong>: Browser control <ul><li>Automates browser tasks like form filling, screenshots, and navigation.</li><li>Executes JavaScript, monitors console logs.</li><li>Seamless integration between LLMs and browsers.</li><li>Also useful for API testing.</li></ul></li></ul></li></ul><h2 id="_12-how-different-roles-use-mcp" tabindex="-1"><a class="header-anchor" href="#_12-how-different-roles-use-mcp" aria-hidden="true">#</a> 12. How Different Roles Use MCP</h2><ul><li>MCP Client Users (e.g., users of Claude Desktop App or Cursor) <ul><li>Just add the desired MCP Server in the config, then simply ask questions as usual</li></ul></li><li>MCP Client Developers (e.g., developers of Claude Desktop App or Cursor) <ul><li>Instantiate the MCP Client</li><li>Establish connection between MCP Client and MCP Server</li><li>For each user query, execute the following logic (tool listing and tool invocation is handled by MCP): <ul><li>Format the user&#39;s query as <code>messages</code></li><li><strong>MCP Client fetches the available tool list (<code>available_tools</code>)</strong></li><li>Calls the LLM with <code>messages</code> and <code>available_tools</code> as input</li><li>The LLM may respond with either: <ul><li><code>text</code>: return the result directly to the user</li><li><code>tool_use</code>: <ul><li><strong>MCP Client invokes the selected tool via the MCP Server</strong></li><li>Get the result of the tool invocation</li><li>Append the original query (<code>role: &quot;user&quot;</code>), LLM response (<code>role: &quot;assistant&quot;</code>), and tool result (<code>role: &quot;user&quot;</code>) into a new <code>messages</code> list</li><li>Call the LLM again with the new <code>messages</code> and <code>available_tools</code></li><li>Return the final response to the user</li></ul></li></ul></li></ul></li></ul></li><li>MCP Server Developers <ul><li>Instantiate an MCP Server</li><li>Define tools and implement their logic</li><li>Run the MCP Server</li></ul></li></ul><figure><img src="'+o+`" alt="MCP Usage Logic" tabindex="0" loading="lazy"><figcaption>MCP Usage Logic</figcaption></figure><h2 id="_13-example-mcp-client-user-experience-claude-desktop-app" tabindex="-1"><a class="header-anchor" href="#_13-example-mcp-client-user-experience-claude-desktop-app" aria-hidden="true">#</a> 13. Example: MCP Client User Experience (Claude Desktop App)</h2><h3 id="_13-1-configure-mcp-server-in-claude-desktop-app" tabindex="-1"><a class="header-anchor" href="#_13-1-configure-mcp-server-in-claude-desktop-app" aria-hidden="true">#</a> 13.1. Configure MCP Server in Claude Desktop App</h3><p>Add a new MCP Server called <code>&quot;weather&quot;</code> under the <code>mcpServers</code> config item</p><ul><li>This tells Claude Desktop App that a server named <code>&quot;weather&quot;</code> is available</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>{
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>After saving, restart Claude Desktop App</p><h3 id="_13-2-view-available-tools-in-claude-desktop-app" tabindex="-1"><a class="header-anchor" href="#_13-2-view-available-tools-in-claude-desktop-app" aria-hidden="true">#</a> 13.2. View Available Tools in Claude Desktop App</h3><p>Users can view all tools provided by all configured MCP Servers, including:</p><ul><li>Tool name</li><li>Description</li><li>Which MCP Server it comes from</li></ul><h3 id="_13-3-ask-related-questions-in-claude-desktop-app" tabindex="-1"><a class="header-anchor" href="#_13-3-ask-related-questions-in-claude-desktop-app" aria-hidden="true">#</a> 13.3. Ask Related Questions in Claude Desktop App</h3><ul><li>The user submits a query</li><li>Claude checks available tools and decides whether and which one to use</li><li>If an MCP tool is selected, the MCP Client executes the tool via the MCP Server</li><li>Execution result is returned to Claude</li><li>Claude converts the result into natural language</li><li>The user receives the response</li></ul><h2 id="_14-example-building-an-mcp-server" tabindex="-1"><a class="header-anchor" href="#_14-example-building-an-mcp-server" aria-hidden="true">#</a> 14. Example: Building an MCP Server</h2><p>The following code creates an MCP Server named <code>&quot;weather&quot;</code> with two tools: <code>get-alerts</code> and <code>get-forecast</code>.</p><h3 id="_14-1-instantiate-the-mcp-server" tabindex="-1"><a class="header-anchor" href="#_14-1-instantiate-the-mcp-server" aria-hidden="true">#</a> 14.1. Instantiate the MCP Server</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>mcp = FastMCP(&quot;weather&quot;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_14-2-define-tools-and-implement-tool-logic" tabindex="-1"><a class="header-anchor" href="#_14-2-define-tools-and-implement-tool-logic" aria-hidden="true">#</a> 14.2. Define Tools and Implement Tool Logic</h3><ul><li>Add <code>@mcp.tool()</code> to define a function as an MCP tool</li><li>No need to manually define an API to list available tools</li></ul><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>@mcp.tool()
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_14-3-run-the-mcp-server" tabindex="-1"><a class="header-anchor" href="#_14-3-run-the-mcp-server" aria-hidden="true">#</a> 14.3. Run the MCP Server</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>mcp.run(transport=&#39;stdio&#39;)
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h2 id="_15-example-building-an-mcp-client" tabindex="-1"><a class="header-anchor" href="#_15-example-building-an-mcp-client" aria-hidden="true">#</a> 15. Example: Building an MCP Client</h2><h3 id="_15-1-create-env-file" tabindex="-1"><a class="header-anchor" href="#_15-1-create-env-file" aria-hidden="true">#</a> 15.1. Create <code>.env</code> File</h3><p>set Anthropic API key</p><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>ANTHROPIC_API_KEY=&lt;your key here&gt;
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div><h3 id="_15-2-define-mcpclient-class" tabindex="-1"><a class="header-anchor" href="#_15-2-define-mcpclient-class" aria-hidden="true">#</a> 15.2. Define <code>MCPClient</code> Class</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    # methods will go here
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-3-implement-server-connection-management-connect-to-server" tabindex="-1"><a class="header-anchor" href="#_15-3-implement-server-connection-management-connect-to-server" aria-hidden="true">#</a> 15.3. Implement Server Connection Management: <code>connect_to_server</code></h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def connect_to_server(self, server_script_path: str):
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-4-implement-query-and-tool-invocation-handler-process-query" tabindex="-1"><a class="header-anchor" href="#_15-4-implement-query-and-tool-invocation-handler-process-query" aria-hidden="true">#</a> 15.4. Implement Query and Tool Invocation Handler: <code>process_query</code></h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def process_query(self, query: str) -&gt; str:
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-5-define-chat-loop-and-cleanup-functions" tabindex="-1"><a class="header-anchor" href="#_15-5-define-chat-loop-and-cleanup-functions" aria-hidden="true">#</a> 15.5. Define <code>chat_loop</code> and <code>cleanup</code> Functions</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def chat_loop(self):
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_15-6-main-entry-point" tabindex="-1"><a class="header-anchor" href="#_15-6-main-entry-point" aria-hidden="true">#</a> 15.6. Main Entry Point</h3><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code>async def main():
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
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,118);function f(C,_){return d(),c("div",null,[h,b,u(" more "),g])}const P=r(p,[["render",f],["__file","036_mcp.html.vue"]]);export{P as default};
