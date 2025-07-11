import searchVDB


import json
from typing import TypedDict, Literal
from langchain.llms.base import LLM
import requests


# === GroqLLM Wrapper ===
class GroqLLM(LLM):
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    api_key: str = "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"  # Replace with your key
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0

    def _call(self, prompt: str, stop: list[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "groq_custom_llm"

# === Instantiate Groq LLM ===
llm = GroqLLM()

# === Graph State Definition ===
class GraphState(TypedDict):
    input: str
    path: Literal["path1", "path2", "path3"]

# === Router Function ===
def router(state: GraphState) -> GraphState:
    prompt = f"""
You are a classifier. Based on the user input, pick exactly one of these paths:

"path1": user asks a general question (no code involved).
"path2": user asks a question and expects code in the answer.
"path3": user provides code and wants it summarized, explained, or fixed.

Respond ONLY with a JSON object exactly like this:

{{"path": "pathX"}}

User input:
{state['input']}
"""
    text = ""
    try:
        result_text = llm.invoke(prompt)
        text = result_text.strip()
        parsed = json.loads(text)
        state["path"] = parsed["path"]
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"LLM output was: {text}")
        state["path"] = "path1"

    print(f"🧭 Chosen path: {state['path']}")
    return state

# === Agents with extra_prompt + RAG context ===
def agent_path1(user_input: str, extra_prompt: str, rag_output: str) -> str:
    prompt = f"{extra_prompt}\n\nContext:\n{rag_output}\n\nUser input:\n{user_input}"
    return llm.invoke(prompt)

def agent_path2(user_input: str, extra_prompt: str, rag_output: str) -> str:
    prompt = f"{extra_prompt}\n\nContext:\n{rag_output}\n\nTask:\n{user_input}"
    return llm.invoke(prompt)

def agent_path3(user_input: str, extra_prompt: str, rag_output: str) -> str:
    prompt = f"{extra_prompt}\n\nContext:\n{rag_output}\n\nCode:\n{user_input}"
    return llm.invoke(prompt)

# === Dispatcher Function ===
def handle_input(user_input: str):
    state: GraphState = {"input": user_input, "path": "path1"}
    state = router(state)

    # 🔹 Placeholder for RAG context (replace with actual index later)
    rag_context = ragout

    # Extra prompts for each path
    extra_prompts = {
        "path1": "You are a helpful assistant that answers general knowledge or conversational questions clearly and concisely.",
        "path2": "You are a code generation assistant. Always provide complete and executable Python code based on the user’s request. Include comments where helpful.",
        "path3": "You are a Python code expert. Explain or fix the provided code. Describe what it does, identify issues, and suggest corrections if needed."
    }

    # Route to correct agent
    if state["path"] == "path1":
        response = agent_path1(user_input, extra_prompts["path1"], rag_context)
    elif state["path"] == "path2":
        response = agent_path2(user_input, extra_prompts["path2"], rag_context)
    elif state["path"] == "path3":
        response = agent_path3(user_input, extra_prompts["path3"], rag_context)
    else:
        response = "Sorry, something went wrong with routing."

    print(f"\n💬 Response:\n{response}")

# === CLI Interface ===
def AgentSystem(UserIn):
    print("🔁 LLM Multi-Agent System (Groq + RAG)\nType 'exit' to quit.")
    user_input = UserIn
    handle_input(user_input)




while True:
    q = input("Enter query (or 'exit' to quit): ")
    if q.lower() == "exit":
        break
    ragout=searchVDB.search_index(searchVDB.index, q)
    AgentSystem(q)