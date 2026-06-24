# LangChain Prompt Templates — Mini Prompt Engine

Solution for the GenAI assignment "Mastering Prompt Templates in LangChain".
Builds dynamic, reusable prompt systems with LangChain (no f-strings, no hardcoding).

## Tasks covered
1. Replace a hardcoded prompt with a reusable `PromptTemplate`
2. Multi-input template (topic, audience, tone)
3. Prompt variations engine (teaching / interview / storytelling)
4. `ChatPromptTemplate` with system + user roles
5. Input validation layer (audience, tone)
6. Prompt generator app (`generate_prompt`)
7. Template reusability test (one template, five inputs)

## Files
- `LangChain_Prompt_Templates.ipynb` — the notebook with all outputs (submit this)
- `LangChain_Prompt_Templates.py`   — same code as a plain script (easy to run in VS Code)

## Setup & run
```bash
pip install -r requirements.txt

# notebook:
jupyter notebook LangChain_Prompt_Templates.ipynb
# or plain script:
python LangChain_Prompt_Templates.py
```
No API key needed — the tasks build and format prompt strings, they do not call an LLM.

> Note: runs on the LangChain 1.x line; prompt classes import from `langchain_core.prompts`
> (the code falls back to the older `langchain.prompts` path automatically).
