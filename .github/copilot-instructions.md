# UltraRAG Copilot Instructions

UltraRAG v2 is a Model Context Protocol (MCP) based RAG framework enabling low-code complex pipeline construction via YAML. **Python 3.11-3.12 required**. Install with `uv pip install -e .` or `pip install -e .`.

## Critical Quick Start

**Before implementing, understand these core concepts:**

1. **MCP Server Registration**: Use `app.tool(method, output="...")` decorator syntax to register tools (see `servers/retriever/src/retriever.py` line 20+ for multi-tool pattern)
2. **Output Mapping**: Format `"input_param->output_key"` maps function parameter to dict key returned. Use commas for multiple outputs: `"q->results,metadata"`
3. **Variable Flow**: Pipeline step outputs automatically available to next step via variable name matching; use `input:` in YAML to explicitly map
4. **Type Hints Required**: Function parameters must have type hints (`str`, `int`, `List[T]`, `Dict[str, Any]`, etc.); returns ALWAYS `Dict[str, ...]`

## Architecture Overview

**Core Pattern: MCP Server + Client + YAML Orchestration**

- **Servers** (in `servers/`): Independent MCP servers wrapping domain functionality (retrieval, generation, evaluation, router, etc.)
  - Each server extends `UltraRAG_MCP_Server` from `src/ultrarag/server.py`
  - Tools registered via `@app.tool(output="...")` decorator (function-level) or `app.tool(method, output="...")` (class-based)
  - Example: `servers/retriever/src/retriever.py` uses class-based registration; `servers/router/src/router.py` uses decorator pattern
  
- **Client** (`src/ultrarag/client.py`): Executes YAML pipelines sequentially via MCP protocol
  - Spawns MCP servers as subprocesses; communicates via stdio (FastMCP handles protocol)
  - Tracks execution state via `UltraData` class (variables, snapshots for debugging)
  - Supports: sequential steps, loops (with memory variables persisted), branches (conditional routing)

- **YAML Pipelines** (in `examples/*.yaml`): Declarative pipeline definition
  - `servers:` section maps server names to `servers/{name}` directories  
  - `pipeline:` section chains tool calls with step execution and control structures
  - Variable flow: Each step's outputs available as variables to subsequent steps
  - **Key examples**: `rag.yaml` (simple), `rag_loop.yaml` (iteration with memory), `IRCoT.yaml` (branches)

## Developer Workflows

### Creating a New Server: Two Patterns

**Pattern 1: Function Decorator (Simple)**
```python
# servers/sayhello/src/sayhello.py
from typing import Dict
from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("sayhello")

@app.tool(output="name->msg")
def greet(name: str) -> Dict[str, str]:
    return {"msg": f"Hello, {name}!"}

if __name__ == "__main__":
    app.run(transport="stdio")
```

**Pattern 2: Class-Based Registration (For Multiple Tools)**
```python
# servers/myserver/src/myserver.py  
class MyServer:
    def __init__(self, app: UltraRAG_MCP_Server):
        app.tool(self.method1, output="input->output")
        app.tool(self.method2, output="data->results,metadata")
    
    def method1(self, input: str) -> Dict[str, str]:
        return {"output": input.upper()}
    
    def method2(self, data: List[str]) -> Dict[str, Any]:
        return {"results": [...], "metadata": {...}}

app = UltraRAG_MCP_Server("myserver")
MyServer(app)

if __name__ == "__main__":
    app.run(transport="stdio")
```

**Key Implementation Details:**
- Output spec `"input_param->output_key"` maps function parameter to dict key (required format)
- Multiple outputs: `"q->results,metadata"` → function `q` param maps to dict keys `results` and `metadata`  
- Return type MUST be `Dict[str, ...]` (FastMCP requirement)
- Optional `servers/{name}/parameter.yaml` for server config; tool signatures auto-generate `servers/{name}/server.yaml`
- Test server in isolation: `python servers/{name}/src/{name}.py` (listens for stdio input from client)

### Building & Running Pipelines

**Build phase** (via `ultrarag build examples/rag.yaml`):
- Discovers all servers, extracts tool metadata via AST inspection
- Generates `parameter/{pipeline}_parameter.yaml` and `server/{pipeline}_server.yaml`
- Validates variable flow between steps
- **AI Tip**: Always run build before run to catch tool registration and naming errors

**Run phase** (via `ultrarag run examples/rag.yaml`):
- Initializes MCP servers as subprocess instances
- Executes pipeline steps sequentially; passes outputs as inputs to next step
- Supports error recovery and cleanup
- **Common issue**: Server crash on startup → check `server.py` for syntax errors or missing imports

**Debugging Pipeline Execution:**
```bash
# See all variable state at each step
LOG_LEVEL=debug ultrarag run examples/pipeline.yaml

# Run specific pipeline only
ultrarag build examples/test.yaml && ultrarag run examples/test.yaml

# Check what tools are discovered
cat examples/server/{pipeline_name}_server.yaml
```

### Control Flow Patterns

**Sequential**: Simple list of tool calls
```yaml
pipeline:
- retriever.search
- generation.generate
```

**Loops**: Iterate steps N times, accumulate results
```yaml
loop:
  times: 3
  steps:
    - retriever.search
    - generation.generate
```

**Branches**: Route execution based on tool outputs
```yaml
branch:
  router:
    - router.check_condition
  branches:
    success_path: [generation.generate]
    failure_path: [custom.handle_error]
```

## Key Files & Patterns

| Path | Purpose |
|------|---------|
| `src/ultrarag/server.py` | `UltraRAG_MCP_Server` base class; `@app.tool()` decorator |
| `src/ultrarag/client.py` | Pipeline executor; build/run async functions; `UltraData` state mgmt |
| `src/ultrarag/cli.py` | CLI entry point; `ultrarag run/build` commands + UI launcher |
| `servers/*/src/*.py` | Individual MCP servers |
| `servers/*/parameter.yaml` | Optional per-server config (hyperparameters, API keys) |
| `examples/*.yaml` | Pipeline definitions |
| `prompt/` | Jinja2 templates for prompt generation |
| `pyproject.toml` | Dependencies: fastmcp==2.11.3 (pin this!), Python 3.11-3.12 |

## Variable Flow & State Management

### Output Mapping (Tool Registration)
- **Decorator format**: `@app.tool(output="param1,param2->key1,key2")` or `@app.tool(output="result")` (if return key name differs)
- Maps function **parameter names** (before arrow) to returned **dict keys** (after arrow)
- Multiple outputs: `"q_ls,top_k->ret_psg"` means `q_ls` param → `ret_psg` key, `top_k` param discarded (no →)
- **Critical rule**: Function signature param name MUST exist; dict return key name comes from output spec

### Input Mapping in YAML Pipelines
- **Auto-flow**: Step outputs automatically flow to next step (variable names match function params)
- **Explicit redirect**: Use `input:` to override: 
  ```yaml
  - retriever.search
  - generation.generate:
      input:
        passages: ret_psg  # Use retriever output 'ret_psg' as 'passages' param
  ```
- **Output rename**: Use `output:` to rename step outputs:
  ```yaml
  - retriever.retriever_search:
      output:
        ret_psg: temp_psg  # Rename ret_psg to temp_psg for next step
  ```

### Memory Variables & Loops
- **Persistence**: Variables prefixed `mem_` or `memory_` persist across loop iterations (see `rag_loop.yaml` line 15+)
- **Accumulation pattern**: 
  ```yaml
  - loop:
      times: 3
      steps:
      - generation.generate
      - custom.accumulate:
          input:
            memory_history: memory_history  # References previous iter's memory_history
  ```

### Branch State Routing
- **Decision variables**: Tools output `branch{depth}_state` to signal conditional execution
- **Router pattern** (see `servers/router/src/router.py`):
  ```python
  @app.tool(output="ans_ls->ans_ls")
  def check_end(ans_ls: List[str]) -> Dict[str, Any]:
      ans_ls = [{"data": ans, "state": "complete" if condition else "incomplete"} for ans in ans_ls]
      return {"ans_ls": ans_ls}
  ```
- **YAML usage**:
  ```yaml
  - branch:
      router: [router.check_end]  # Executes router tool, sets branch1_state
      branches:
        complete: []              # Exec if branch1_state == "complete"
        incomplete: [...]         # Exec if branch1_state == "incomplete"
  ```

Example from `IRCoT.yaml` (complex loop + branch):
```yaml
- loop:
    times: 2
    steps:
    - generation.generate
    - branch:
        router: [router.ircot_check_end]
        branches:
          incomplete: [custom.ircot_process]  # Continue if incomplete
          complete: []                         # Exit loop if complete
```

## Critical Implementation Details

### Async Execution Model
- **Client** (`src/ultrarag/client.py`): Uses `asyncio` for concurrent server management
  - `async def build()`: Discovers and validates all servers, generates metadata files
  - `async def run()`: Sequentially executes pipeline steps; await each tool call
  - `async with client:` ensures proper server lifecycle (startup/cleanup)
- **Servers**: Each runs as isolated subprocess; FastMCP handles stdio communication
  - Tool execution is async-capable but steps execute sequentially within a pipeline
  - Example server lifecycle:
    ```python
    async def run(self, transport: Transport = "stdio"):
        # Server starts, listens for MCP calls, auto-shuts down after client disconnects
        await self.lifespan()
    ```

### Tool Registration & Metadata
1. `@app.tool()` must be called on function definition; **changes require server restart**
2. Output spec format: `"input_param->output_key"` or `"output_key"` (if no input mapping)
   - Maps function parameter name to returned dict key
   - Multiple outputs: `"result->ans,confidence"` → `{"ans": ..., "confidence": ...}`
3. Parameter resolution: All step inputs resolved before execution via `_resolve_vars()` in client
4. Type hints: Function parameters auto-converted to tool JSON schema; **Dict return always assumed**
5. Server discovery: Via `FastMCP` introspection of decorated functions; tool names = function names

### Type Constraints & Validation
- **Supported parameter types**: `str`, `int`, `float`, `bool`, `List[T]`, `Dict[str, Any]`, `Optional[T]`
  - Type hints are **required** for proper JSON schema generation
  - Union types should use `Optional[T]` pattern; complex unions not supported
  - Default values enable optional parameters in YAML calls
  
- **Return type requirement**: **Always `Dict[str, ...]`** - scalar returns are NOT converted
  ```python
  # ✅ Correct
  def process(query: str) -> Dict[str, str]:
      return {"result": query.upper()}
  
  # ❌ Wrong - will fail
  def process(query: str) -> str:
      return query.upper()
  ```

- **Type validation in YAML**: FastMCP performs JSON schema validation on inputs
  - Mismatched types cause tool execution to fail with schema error
  - Use `List[Dict[str, Any]]` for heterogeneous data, not plain `List`
  
- **Datetime & Complex Objects**: Not natively supported; serialize to ISO string (`str`)
  ```python
  # ✅ Correct
  @app.tool(output="timestamp->result")
  def get_data(timestamp: str) -> Dict[str, str]:  # ISO format: "2025-01-01T00:00:00"
      dt = datetime.fromisoformat(timestamp)
      return {"result": str(dt)}
  ```

### Server Communication Pattern
```python
# Server-side (servers/myserver/src/myserver.py)
from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server("myserver")

@app.tool(output="query->results")
async def search(query: str, top_k: int = 5) -> Dict[str, List[str]]:
    """Tool receives inputs, returns dict with output keys."""
    results = await some_async_op(query)
    return {"results": results[:top_k]}

if __name__ == "__main__":
    app.run(transport="stdio")  # Subprocess communicates via stdin/stdout with client
```

```python
# Client-side (src/ultrarag/client.py - simplified)
async with client:
    # Client calls server tools via MCP protocol
    result = await client.call_tool("myserver.search", {
        "query": "example",
        "top_k": 10
    })
    # Returns: {"results": [...]}, accessible in next step as step variable
```

## Error Handling & Recovery

### Server-Level Error Handling
```python
# servers/retriever/src/retriever.py
from ultrarag.server import UltraRAG_MCP_Server
app = UltraRAG_MCP_Server("retriever")

@app.tool(output="query->results")
def search(query: str) -> Dict[str, Any]:
    try:
        results = execute_search(query)
        if not results:
            app.logger.warning(f"No results for query: {query}")
            return {"results": [], "status": "no_results"}
        return {"results": results, "status": "success"}
    except Exception as e:
        app.logger.error(f"Search failed: {str(e)}")
        raise  # Re-raise to fail the pipeline step (caught by client)
```

### Pipeline-Level Error Handling
- **Execution halts on tool exception** unless branch/loop has fallback path
- Use `router` server to check intermediate results and route to error handlers:
```yaml
pipeline:
- retriever.search
- branch:
    router:
      - router.check_results  # Outputs: branch_state = "success" or "failure"
    branches:
      success:
        - generation.generate
      failure:
        - custom.handle_empty_results  # Fallback logic
```

- **Snapshots** (in `UltraData`): Each step creates snapshot of variables for debugging
  - Access via: `Data.snapshots[step_index]` to inspect variable state at any point

## Common Tasks & Examples

### Task 1: Debug a Tool Call
```python
# servers/debug-example/src/debug_example.py
@app.tool(output="input->output")
def process(input: str) -> Dict[str, str]:
    app.logger.info(f"Input received: {input}")  # Use structured logging
    
    intermediate = transform(input)
    app.logger.debug(f"After transform: {intermediate}")
    
    result = final_process(intermediate)
    app.logger.info(f"Final output: {result}")
    
    return {"output": result}
```
**Run pipeline with logs**: `ultrarag run examples/pipeline.yaml 2>&1 | grep -E "DEBUG|ERROR|INFO"`

### Task 2: Add New Evaluation Metric
```python
# servers/evaluation/src/evaluation.py
@app.tool(output="predictions,references->scores")
def evaluate_custom(
    predictions: List[str],
    references: List[str],
    metric: str = "exact_match"
) -> Dict[str, Any]:
    """Custom evaluation metric added to existing evaluation server."""
    scores = []
    for pred, ref in zip(predictions, references):
        if metric == "exact_match":
            score = 1.0 if pred.strip() == ref.strip() else 0.0
        elif metric == "partial":
            score = len(set(pred.split()) & set(ref.split())) / max(len(set(ref.split())), 1)
        scores.append(score)
    
    return {
        "scores": scores,
        "avg_score": sum(scores) / len(scores) if scores else 0.0
    }
```
**Usage in YAML**: Add `- evaluation.evaluate_custom` step, parameters set via `servers/evaluation/parameter.yaml`

### Task 3: Custom Variable Transformation
```python
# servers/custom/src/custom.py - transforms incompatible outputs
@app.tool(output="source_output->target_format")
def transform_output(source_output: Dict[str, Any]) -> Dict[str, Any]:
    """Convert retriever results to generation input format."""
    passages = source_output.get("results", [])
    
    # Reshape for generation model
    formatted_passages = [
        {"text": p, "rank": i}
        for i, p in enumerate(passages)
    ]
    
    return {"target_format": formatted_passages}
```
**YAML usage**: Insert transformation step between incompatible services
```yaml
pipeline:
- retriever.search         # Output: {"results": [...]}
- custom.transform_output  # Transform to gen-compatible format
- generation.generate      # Input: {"passages": [...]}
```

### Task 4: Profile Pipeline Performance
```python
# Run with timing info
ultrarag run examples/rag.yaml --log-level=debug 2>&1 | grep "Step timing"

# Or add custom timing in client loop (src/ultrarag/client.py):
import time
start = time.time()
result = await client.call_tool(tool_name, kwargs)
elapsed = time.time() - start
logger.info(f"Tool {tool_name} took {elapsed:.2f}s")
```

### Task 5: Handle Variable State Across Loops
```python
# servers/router/src/router.py - accumulate state across iterations
@app.tool(output="current_result,memory_history->memory_history,branch_state")
def check_and_accumulate(
    current_result: str,
    memory_history: List[str] = None
) -> Dict[str, Any]:
    """Accumulate results and decide loop termination."""
    if memory_history is None:
        memory_history = []
    
    memory_history.append(current_result)
    
    # Decision logic
    if len(memory_history) >= 3 or is_high_quality(current_result):
        branch_state = "complete"
    else:
        branch_state = "incomplete"
    
    return {
        "memory_history": memory_history,  # Persisted across loop
        "branch_state": branch_state
    }
```
**YAML pattern**:
```yaml
- loop:
    times: 5
    steps:
    - generation.generate
    - router.check_and_accumulate:
        input:
          memory_history: memory_history  # Reference previous memory
    - branch:
        router: [router.check_and_accumulate]
        branches:
          complete: []
          incomplete: []
```

### Task 6: Multi-Server Integration Pattern
```yaml
# examples/complex_pipeline.yaml - chaining multiple specialized servers
servers:
  corpus: servers/corpus        # Knowledge base preparation
  retriever: servers/retriever  # Retrieval layer
  reranker: servers/reranker    # Ranking layer
  generation: servers/generation
  evaluation: servers/evaluation

pipeline:
- corpus.build_index           # Initialize knowledge base
- retriever.retriever_init
- retriever.search             # Get candidate passages
- reranker.rerank:             # Re-rank candidates
    input:
      query: query
      passages: search_results
- generation.generate:         # Use top-ranked passages
    input:
      passages: reranked_passages
      query: query
- evaluation.evaluate
```

## Configuration & Testing

### Installation & Requirements
```bash
# Core installation
uv pip install -e .

# With optional dependencies
uv pip install -e ".[all]"

# Individual feature sets
uv pip install -e ".[retriever]"    # Search backends (BM25, FAISS, Infinity)
uv pip install -e ".[generation]"   # LLM backends (vLLM, OpenAI, HF)
uv pip install -e ".[corpus]"       # Document processing (MinerU, PyMuPDF)
```

**Critical Requirements**: 
- Python 3.11–3.12 (3.13+ not supported)
- FastMCP 2.11.3 (pin this version!)
- Node.js installed (for MCP protocol support; check with `node --version`)

**Verify Installation**:
```bash
ultrarag run examples/sayhello.yaml  # Shows "Hello, UltraRAG 2.0!"
```

### Testing Pattern: Three-Level Approach

**Level 1: Server Unit Tests** (test tool logic in isolation)
```python
# servers/myserver/tests/test_myserver.py
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from myserver import app

@pytest.mark.asyncio
async def test_tool_output_shape():
    """Verify tool returns correct dict structure."""
    result = await app.my_tool(input="test")
    assert isinstance(result, dict)
    assert "output_key" in result  # Must match output= spec

# Run: pytest servers/myserver/tests/ -v
```

**Level 2: Build Validation** (check tool discovery & metadata generation)
```bash
# Build extracts all tool signatures and creates server.yaml
ultrarag build examples/test_pipeline.yaml

# Verify tools exist in generated metadata
grep "my_tool" examples/server/test_pipeline_server.yaml

# Check generated parameter spec
cat examples/parameter/test_pipeline_parameter.yaml
```

**Level 3: Pipeline Integration** (test end-to-end execution)
```bash
# Minimal test pipeline: examples/test_simple.yaml
servers:
  custom: servers/custom
pipeline:
- custom.my_tool

# Execute with verbose logging
LOG_LEVEL=debug ultrarag run examples/test_simple.yaml
```

### Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ValidationError: Tool not found` | Server not registered or tool name mismatch | Check `@app.tool()` decorator exists, function name matches YAML |
| `TypeError: ... required positional argument` | Missing required input parameter | Add `input:` mapping in YAML or check function defaults |
| `KeyError: output_key` | Output dict doesn't have expected key | Verify return dict matches `output=` spec (before arrow = param name, after = dict key) |
| Server won't start | Import error or syntax error in server | Run `python servers/{name}/src/{name}.py` directly to see error |
| `AttributeError: dict has no attribute 'get'` | Type mismatch in tool input | Ensure YAML passes correct type (dict → `Dict[str, Any]`, list → `List[T]`) |

## Advanced Patterns

### Working with External MCP Servers
```yaml
# Use non-UltraRAG MCP servers (Node.js, external services)
servers:
  custom_search: "/path/to/external/mcp_server.js"
  builtin_retriever: servers/retriever

pipeline:
- custom_search.query_tool
- builtin_retriever.rerank
```

### Conditional Input Resolution
```yaml
# Use different inputs based on previous step output
- retriever.search
- custom.check_quality:
    input:
      passages: results  # Auto-resolved from previous step output
      threshold: 0.5     # Static value
- generation.generate:
    input:
      query: query  # Can reference from multiple steps back
```

### Persisting State Across Pipeline Runs
```python
# servers/custom/src/custom.py
import json
from pathlib import Path

@app.tool(output="state->state")
def save_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Persist state to disk for resuming interrupted pipelines."""
    state_file = Path("./pipeline_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f)
    app.logger.info(f"State saved to {state_file}")
    return {"state": state}

@app.tool(output="->state")
def load_state() -> Dict[str, Any]:
    """Resume from saved checkpoint."""
    state_file = Path("./pipeline_state.json")
    if state_file.exists():
        with open(state_file) as f:
            return {"state": json.load(f)}
    return {"state": {}}
```

## Performance Optimization & Best Practices

### Memory Management

**Avoid loading entire datasets into memory**:
```python
# ❌ Wrong - loads all results at once
@app.tool(output="query->results")
def search_all(query: str) -> Dict[str, List[str]]:
    all_results = fetch_from_database(query)  # Could be millions
    return {"results": all_results}

# ✅ Correct - use pagination or streaming
@app.tool(output="query->results")
def search_paginated(query: str, page: int = 0, per_page: int = 100) -> Dict[str, Any]:
    results = fetch_from_database(query, offset=page*per_page, limit=per_page)
    return {
        "results": results,
        "page": page,
        "has_next": len(results) == per_page
    }
```

**Reuse expensive resources**:
```python
# Cache embeddings model or retrieval index
_cached_model = None

@app.tool(output="query->embedding")
def embed(query: str) -> Dict[str, List[float]]:
    global _cached_model
    if _cached_model is None:
        _cached_model = load_expensive_model()  # Load once
        app.logger.info("Model loaded into memory")
    
    embedding = _cached_model.encode(query)
    return {"embedding": embedding.tolist()}
```

**Stream large outputs**:
```python
# For large result sets, consider compression or chunking
@app.tool(output="query->results")
def search_streaming(query: str) -> Dict[str, Any]:
    results = search_database(query)
    
    # Truncate if too large
    if len(results) > 10000:
        app.logger.warning(f"Truncating {len(results)} results to 10000")
        results = results[:10000]
    
    return {
        "results": results,
        "total_count": len(results),
        "truncated": len(results) > 10000
    }
```

### Async Best Practices

**Use async I/O for network/database calls**:
```python
import asyncio
import aiohttp

@app.tool(output="urls->responses")
async def fetch_parallel(urls: List[str]) -> Dict[str, Any]:
    """Fetch multiple URLs concurrently using async."""
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            results.append({"url": urls[i], "error": str(resp)})
        else:
            results.append({"url": urls[i], "status": resp.status})
    
    return {"responses": results}
```

**Avoid blocking operations in async functions**:
```python
# ❌ Wrong - blocks the event loop
@app.tool(output="query->result")
async def process_wrong(query: str) -> Dict[str, str]:
    result = expensive_cpu_operation(query)  # BLOCKS!
    return {"result": result}

# ✅ Correct - use asyncio or threadpool for CPU-bound work
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@app.tool(output="query->result")
async def process_correct(query: str) -> Dict[str, str]:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, expensive_cpu_operation, query)
    return {"result": result}
```

**Set timeouts for long-running operations**:
```python
@app.tool(output="query->results")
async def search_with_timeout(query: str, timeout_secs: int = 30) -> Dict[str, Any]:
    try:
        results = await asyncio.wait_for(
            search_database_async(query),
            timeout=timeout_secs
        )
        return {"results": results, "timeout": False}
    except asyncio.TimeoutError:
        app.logger.warning(f"Search timed out after {timeout_secs}s")
        return {"results": [], "timeout": True}
```

### Pipeline-Level Performance

**Minimize data copying between steps**:
```yaml
# In YAML pipelines, avoid intermediate transformation steps when possible
pipeline:
- retriever.search
- generation.generate:
    input:
      passages: results  # Direct reference, no copy
```

**Batch operations when feasible**:
```python
# servers/custom/src/custom.py
@app.tool(output="queries->results")
def batch_search(queries: List[str]) -> Dict[str, List[Any]]:
    """Search multiple queries in one call, more efficient than loop."""
    batch_size = 32
    all_results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_results = search_engine.search_batch(batch)
        all_results.extend(batch_results)
    
    return {"results": all_results}
```

**Use memory variables to accumulate across loops**:
```yaml
# Good: accumulates results efficiently
loop:
  times: 5
  steps:
  - retriever.search
  - custom.accumulate:
      input:
        current: results
        memory_history: memory_history  # Persists across iterations
```

### Monitoring & Profiling

**Add timing instrumentation**:
```python
import time

@app.tool(output="query->results")
def search_timed(query: str) -> Dict[str, Any]:
    start = time.perf_counter()
    
    results = search_database(query)
    
    elapsed = time.perf_counter() - start
    app.logger.info(f"Search took {elapsed:.3f}s")
    
    return {
        "results": results,
        "query_time_ms": int(elapsed * 1000)
    }
```

**Log memory usage**:
```python
import psutil
import os

@app.tool(output="query->results")
def search_monitored(query: str) -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    results = search_database(query)
    
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_delta = mem_after - mem_before
    
    app.logger.info(f"Memory delta: {mem_delta:.1f}MB, current: {mem_after:.1f}MB")
    
    return {"results": results}
```
