Here is the architecture for the Medici Project.



### 1. Pre-Processing & Tool Selection
* **Data Ingestion:** Python pulls end-of-day (EOD) market data.
* **Dependency Mapping:** Python parses your master list of indicator tools, explicitly flagging overlaps and dependencies.
* **Chunked Scoring (Map-Reduce[^1]):** Qwen 4B loads a specific persona profile, reads the tool list in discrete chunks, and scores their utility. VRAM is flushed to disk after every chunk.
* **Execution:** Python executes only the high-scoring, non-redundant tools. Results are packed into a minified JSON payload.

### 2. Historical Context (RAG[^2])
* The fresh JSON payload is embedded into a vector.
* The local vector database retrieves the 3-5 most mathematically similar past trades, including their initial setups, agent reasoning, and final profit/loss outcomes.

### 3. Adversarial Pipeline (Sequential Execution)
To fit within your hardware limits, isolated agents run sequentially. State is written to your local drive and VRAM is wiped between each step.
* **Isolated Generation:** Agent A and Agent B independently analyze the JSON and historical context to form their initial theses.
* **Cross-Check:** Agents review each other's isolated outputs strictly to identify logical fallacies and data misinterpretations (Red Teaming[^3]).

### 4. Synthesis & Human-in-the-Loop (HITL)
* **The Judge:** A final local agent synthesizes the isolated theses and cross-checks into a formatted EOD report.
* **Terminal Interrupt:** The Python script pauses in your nvim environment, waiting for manual approval or a follow-up query.
* **Semantic Routing:** If you ask a question, the Judge acts purely as a router, determining which specific sub-agent must answer.
* **Targeted Context:** Only the targeted agent's prior thesis and your new question are loaded into Qwen to answer the query, preventing context window bloat.

### 5. Escalation & Validation
* **Opus Handoff:** For complex setups, the locally synthesized context package is sent via API to Claude 3 Opus for deep macroeconomic reasoning and final trade determination.
* **Validation:** The entire system logic is validated strictly through chronological walk-forward testing. No tweaking rules to pass failed past windows.

***

[^1]: **Map-Reduce**: Splitting a large dataset into smaller chunks for sequential processing, then combining the results into a filtered output.
[^2]: **RAG (Retrieval-Augmented Generation)**: Fetching historical data from an external database to provide context for an AI's current analysis.
[^3]: **Red Teaming**: Using an independent agent to ruthlessly attack a system or thesis to expose vulnerabilities.
