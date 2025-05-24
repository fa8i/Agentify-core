from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from openai import AzureOpenAI, RateLimitError, OpenAI
from agentify.client_builder import LLMClientFactory, LLMClientType


@dataclass(slots=True)
class Tool:
    """Wrapper que vincula un JSON-schema con una función Python."""
    schema: Dict[str, Any]
    func: Callable[..., Any]

    def __post_init__(self) -> None:
        if "name" not in self.schema:
            raise ValueError("Tool schema must include 'name'")

    @property
    def name(self) -> str:
        return self.schema["name"]

    def __call__(self, **kwargs: Any) -> str:
        """Ejecuta la función y devuelve JSON o string; captura errores genéricos."""
        try:
            result = self.func(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)


class BaseAgent:
    """Agente autocontenido con function-calling, herramientas y memoria."""
    MAX_TOOL_ITER = 5
    RETRIES = 3

    def __init__(
        self,
        name: str,
        system_prompt: str,
        provider: str,
        client_factory: LLMClientFactory = LLMClientFactory(),
        model_name: str | None = None,
        temperature: Optional[float] = None,
        tools: List[Tool] | None = None,
        client_config_override: Optional[Dict[str, Any]] = None,
        agent_timeout: Optional[int] = 30,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self._tools: Dict[str, Tool] = {t.name: t for t in tools or []}
        self._tool_defs = [
            {"type": "function", "function": t.schema} for t in self._tools.values()
        ]
        self.client: LLMClientType = client_factory.create_client(
            provider=self.provider,
            config_override=client_config_override,
            timeout=agent_timeout
        )
        self._history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)
    
    @property
    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def add(self, role: str, content: Optional[str], **kwargs: Any) -> None:
        msg: Dict[str, Any] = {"role": role}
        if content:
            msg["content"] = content
        msg.update(kwargs)
        self._history.append(msg)

    def clear_memory(self) -> None:
        self._history = [{"role": "system", "content": self.system_prompt}]

    def save_history(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)

    def load_history(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._history = json.load(f)

    def _completion(self) -> Any:
        tool_choice_param = "auto" if (tools_param := self._tool_defs) else None
        for retry in range(self.RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self._history,
                    tools=tools_param,
                    tool_choice=tool_choice_param,
                    temperature=self.temperature
                )
                return response.choices[0].message
            except RateLimitError:
                time.sleep(2 ** retry)
            except Exception as e:
                print(f"Error during completion (attempt {retry+1}/{self.RETRIES}): {e}")
                if retry == self.RETRIES - 1:
                    raise
                time.sleep(2 ** retry) # Esperar antes de reintentar por otros errores también

        raise RuntimeError(f"{self.client.__class__.__name__} completion failed after retries")


    def respond(self, user_input: str) -> str:
        self.add("user", user_input)
        assistant_parts: List[str] = []

        for _iteration in range(self.MAX_TOOL_ITER):
            msg = self._completion()
            if current_turn_content := msg.content:
                assistant_parts.append(current_turn_content)

            if not msg.tool_calls:
                if msg.content:
                    self.add("assistant", msg.content)
                break
            
            # Solo añade el mensaje parcial si hay contenido o tool_calls
            tool_call_message = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
                        
            self.add(**tool_call_message)

            # ejecutar tools
            for tc in msg.tool_calls:
                tool = self._tools.get(tc.function.name)
                if not tool:
                    tool_result = json.dumps({"error": f"tool {tc.function.name} not registered"})
                else:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as exc:
                        tool_result = json.dumps({"error": f"bad args - {exc}"})
                    else:
                        tool_result = tool(**args)

                self.add("tool", tool_result, tool_call_id=tc.id, function_name=tc.function.name)
        else:
            pass  

        return "\n\n".join(assistant_parts).strip("\n")
    
    def tool_exists(self, name: str) -> bool:
        return name in self._tools

    def unregister_tool(self, name: str) -> bool:
        if name not in self._tools:
            return False
        self._tools.pop(name)
        self._tool_defs = [d for d in self._tool_defs if d["function"]["name"] != name]
        return True

    def register_tool(self, tool: Tool) -> None:
        self.unregister_tool(tool.name)  # si existe, actualiza la tool
        self._tools[tool.name] = tool
        self._tool_defs.append({"type": "function", "function": tool.schema})



class AgentRouter:
    """Mini-bus para conversaciones entre múltiples agentes."""

    def __init__(self, *agents: BaseAgent):
        self.agents = {a.name: a for a in agents}

    def converse(self, sender: str, receiver: str, message: str, echo: bool = False) -> str:
        if sender not in self.agents or receiver not in self.agents:
            raise KeyError("Unknown agent in conversation")

        if echo:
            print(f"[{sender}\u27a4{receiver}]: {message}")

        response = self.agents[receiver].respond(message)
        if echo:
            print(f"[{receiver}]: {response}")

        return response
