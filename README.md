# LiteLLM Agent Components

LangFlow agent components that connect to a LiteLLM proxy server for multi-provider LLM support. Two implementations are available with different capabilities.
This is an AI assisted rewrite from the original Agent component to better leverage the unique features of the LiteLLM API, including provider-specific metadata extraction and enhanced response handling.

## Component Variants

### 1. LiteLLM/litellm_agent_stream.py - Basic Agent

Simple, standard agent implementation using LangChain's ChatOpenAI with streaming mode.

**Features:**
- Full tool calling support with automatic callback handling
- Optional built-in current date tool
- Model parameters pass-through for provider-specific settings
- Standard streaming mode for real-time token output

**Use this if:**
- You need basic agent functionality
- Streaming responses are acceptable
- You don't require response metadata

### 2. LiteLLM/litellm_agent.py - Enhanced Agent with Metadata Extraction

Advanced agent implementation with non-streaming mode to preserve provider metadata (citations, search results, usage info, etc.).

**Features:**
- All basic features plus:
- **Non-streaming mode** forces `stream=False` to preserve response metadata
- **Dynamic property extraction**: Extract any top-level response properties from the LiteLLM API and append them to the response
- Metadata automatically formatted with XML-like tags for easy parsing

**Use this if:**
- You don't need streaming responses
- You need provider-specific metadata from responses

## Configuration

### Required Inputs (Both Versions)

| Input | Description | Example |
|-------|-------------|---------|
| `base_url` | LiteLLM proxy base URL | `http://localhost:8000` |
| `api_key` | API key for the LiteLLM proxy | `sk-...` |
| `model_name` | Model identifier | `gpt-4`, `claude-3-5-sonnet`, `perplexity/sonar` |

### Optional Inputs (Both Versions)

| Input | Description | Default |
|-------|-------------|---------|
| `system_prompt` | Agent instructions | "You are a helpful assistant..." |
| `input` | User input for the agent | "What is the capital of France?" |
| `tools` | LangChain tools for the agent to use | `[]` |
| `add_current_date_tool` | Enable current date tool | `true` |
| `model_params` | Provider-specific parameters | `{}` |

### Enhanced-Only Input

| Input | Description | Example |
|-------|-------------|---------|
| `extract_top_level_properties` | Comma-separated list of response properties to extract and append | `citations,search_results` or `cost,usage` |

## Property Extraction (Enhanced Version)

The enhanced agent allows you to extract any top-level property from the LiteLLM API response. These properties are automatically appended to the response content wrapped in XML-like tags.

**Example with Perplexity:**
```
Input: "What happened in the news today?"
Model: perplexity/sonar
Extract: citations,search_results

Output:
[Model response text]

<citations>
[
  {"title": "Article Title", "url": "https://..."},
  ...
]
</citations>

<search_results>
[
  {"title": "Result 1", ...},
  ...
]
</search_results>
```

**Common properties to extract:**
- `citations` - Source attributions (Perplexity)
- `search_results` - Search result details (Perplexity)
- `usage` - Token usage information
- `cost` - Estimated API cost
- Custom provider-specific fields
