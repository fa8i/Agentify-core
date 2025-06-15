import logging
import json
import time
from typing import Any, Dict, Generator, List, Optional, Union
import uuid

from openai import RateLimitError
from agentify.core import BaseAgent, Tool
from agentify.client_builder import LLMClientFactory, LLMClientType
from agentify.events import (
    AgentFinishData,
    AgentStartData,
    ErrorData,
    StreamingEvent,
    ToolCallData,
    ToolResultData,
    WarnData,
)

logger = logging.getLogger(__name__)

class WorkerAgent(BaseAgent):
    """
    Clase del núcleo de agentes de IA agnóstico a frameworks. (@fa8i)
    """

    MAX_TOOL_ITER: int = 5
    RETRIES: int = 3

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        provider: str,
        model_name: str,
        temperature: Optional[float] = 0.7,
        tools: Optional[List[Tool]] = None,
        client_factory: LLMClientFactory = LLMClientFactory(),
        client_config_override: Optional[Dict[str, Any]] = None,
        agent_timeout: Optional[int] = 60,
        stream: bool = False,
        memory_type: str = "in_memory",
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, description, system_prompt, memory_type, memory_config)

        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self._tools: Dict[str, Tool] = {}
        self._tool_defs: List[Dict[str, Any]] = []
        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        self.stream: bool = stream
        self.client: LLMClientType = client_factory.create_client(
            provider=self.provider,
            config_override=client_config_override,
            timeout=agent_timeout,
        )

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Devuelve una copia del historial de conversación desde la memoria."""
        return self.memory.get_history()

    @property
    def list_tools(self) -> List[str]:
        """Devuelve los nombres de las herramientas registradas."""
        return list(self._tools.keys())

    def add(self, role: str, content: Optional[str] = None, **kwargs: Any) -> None:
        """Añade un mensaje a la memoria del agente."""
        msg: Dict[str, Any] = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(kwargs)  # Usado para tool_calls, tool_call_id, name (rol 'tool')
        self.memory.add(msg)

    def clear_memory(self) -> None:
        """Reinicia la memoria."""
        self.memory.clear()

    def save_history(self, path: str) -> None:
        """Guarda el historial en un archivo JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)

    def load_history(self, path: str) -> None:
        """Carga el historial desde un archivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            self._history = json.load(f)

    def _completion(self) -> Union[Any, Generator[Dict[str, Any], None, None]]:
        """Realiza la llamada al LLM, con reintentos y manejo de errores."""
        tool_choice_param = "auto" if self._tool_defs else None
        common_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self.memory.get_history(),
            "temperature": self.temperature,
        }
        if self._tool_defs:  # Solo añadir tools y tool_choice si hay herramientas
            common_params["tools"] = self._tool_defs
            common_params["tool_choice"] = tool_choice_param

        for attempt in range(self.RETRIES):
            try:
                if self.stream:
                    return self.client.chat.completions.create(
                        **common_params, stream=True
                    )
                response = self.client.chat.completions.create(
                    **common_params, stream=False
                )
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message
                raise ValueError(
                    "La respuesta de la API no contenía 'choices' válidos."
                )
            except RateLimitError:
                if attempt == self.RETRIES - 1:
                    logger.error(
                        "Límite de tasa de API alcanzado después de reintentos."
                    )
                    raise
                sleep_time = 2**attempt
                logger.warning(
                    f"Límite de tasa alcanzado. Reintentando en {sleep_time}s..."
                )
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(
                    f"Error en _completion (intento {attempt + 1}/{self.RETRIES}): {e}",
                    exc_info=True,
                )
                if attempt == self.RETRIES - 1:
                    raise
                time.sleep(2**attempt)

        msg = f"La completación del LLM ({self.client.__class__.__name__}) falló después de {self.RETRIES} reintentos."
        logger.critical(msg)
        raise RuntimeError(msg)

    def _split_concatenated_json_objects(self, json_string: str) -> List[str]:
        """Intenta dividir una cadena que puede contener múltiples JSONs concatenados."""
        objects_str: List[str] = []
        decoder = json.JSONDecoder()
        original_string_trimmed = json_string.strip()
        pos = 0

        if not original_string_trimmed:
            return []
        try:
            json.loads(original_string_trimmed)
            return [original_string_trimmed]  # Es un JSON único válido
        except json.JSONDecodeError:
            pass  # No es un JSON único, intentar dividir

        while pos < len(original_string_trimmed):
            try:
                _, consumed_chars = decoder.raw_decode(original_string_trimmed[pos:])
                objects_str.append(original_string_trimmed[pos : pos + consumed_chars])
                pos += consumed_chars
                while (
                    pos < len(original_string_trimmed)
                    and original_string_trimmed[pos].isspace()
                ):
                    pos += 1
            except json.JSONDecodeError:
                if not objects_str:  # No se pudo parsear nada desde el inicio
                    logger.warning(f"No se pudo decodificar JSON de: '{json_string}'")
                    return [
                        json_string
                    ]  # Devolver original para que falle el parseo más adelante
                logger.warning(
                    f"Agente '{self.name}': No se pudo decodificar más JSON en la posición {pos} de '{original_string_trimmed}'. Objetos parseados: {len(objects_str)}."
                )
                break
        return objects_str if objects_str else [json_string]

    def _process_agent_logic(self, user_input: str) -> Generator[StreamingEvent, None, None]:
        """
        Generador interno que maneja la lógica del agente y emite eventos de streaming estructurados.
        """
        # Emitir evento de inicio del agente
        yield StreamingEvent(
            event_type="AGENT_START",
            source_agent=self.name,
            data=AgentStartData(user_input=user_input),
        )
        self.add(role="user", content=user_input)

        final_content = ""

        try:
            for iteration_count in range(self.MAX_TOOL_ITER):
                response_or_stream = self._completion()

                current_turn_content_parts: List[str] = []
                assembled_tool_calls: List[Dict[str, Any]] = []

                if self.stream:
                    if not (hasattr(response_or_stream, "__iter__") and hasattr(response_or_stream, "__next__")):
                        raise TypeError(f"Se esperaba un iterador en modo streaming, se obtuvo {type(response_or_stream)}.")

                    tool_call_assembler: Dict[int, Dict[str, Any]] = {}
                    for chunk in response_or_stream:
                        if not chunk.choices: 
                            continue
                        delta = chunk.choices[0].delta

                        if delta.content:
                            yield StreamingEvent(
                                event_type="LLM_CHUNK",
                                source_agent=self.name,
                                data=delta.content
                            )
                            current_turn_content_parts.append(delta.content)

                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_call_assembler:
                                    tool_call_assembler[idx] = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                                call_data = tool_call_assembler[idx]
                                if tc_delta.id and not call_data["id"]: 
                                    call_data["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name: 
                                        call_data["function"]["name"] = tc_delta.function.name
                                    if tc_delta.function.arguments: 
                                        call_data["function"]["arguments"] += tc_delta.function.arguments
                    
                    for idx in sorted(tool_call_assembler.keys()):
                        call_data = tool_call_assembler[idx]
                        if not call_data.get("id"): 
                            call_data["id"] = f"s_{self.provider[:3]}_tc_{iteration_count}_{idx}_{uuid.uuid4().hex[:6]}"
                        if call_data.get("function", {}).get("name"): 
                            assembled_tool_calls.append(call_data)
                
                else:  # Modo no-streaming
                    msg_object = response_or_stream
                    if not (hasattr(msg_object, "content") or hasattr(msg_object, "tool_calls")):
                        raise TypeError(f"Se esperaba un objeto Message en modo no-streaming, se obtuvo {type(msg_object)}")

                    if msg_object.content:
                        yield StreamingEvent(event_type="LLM_CHUNK", source_agent=self.name, data=msg_object.content)
                        current_turn_content_parts.append(msg_object.content)

                    if msg_object.tool_calls:
                        for i, tc in enumerate(msg_object.tool_calls):
                            tc_id = tc.id or f"ns_{self.provider[:3]}_tc_{iteration_count}_{i}_{uuid.uuid4().hex[:6]}"
                            assembled_tool_calls.append({"id": tc_id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}})

                # Procesamiento y expansión de Tool Calls
                expanded_tool_calls: List[Dict[str, Any]] = []
                if assembled_tool_calls:
                    for tc_original in assembled_tool_calls:
                        tool_name = tc_original.get("function", {}).get("name", "unknown_tool")
                        args_value = tc_original.get("function", {}).get("arguments")
                        args_str = args_value if isinstance(args_value, str) else json.dumps(args_value or {})
                        original_id = tc_original.get("id", f"gen_id_{uuid.uuid4().hex[:4]}")
                        if not args_str.strip():
                            tc_original["function"]["arguments"] = "{}"
                            expanded_tool_calls.append(tc_original)
                            continue
                        split_args_json = self._split_concatenated_json_objects(args_str)
                        if len(split_args_json) > 1:
                            for i, single_arg_json in enumerate(split_args_json):
                                expanded_tool_calls.append({"id": f"{original_id}_part_{i}", "type": "function", "function": {"name": tool_name, "arguments": single_arg_json}})
                        elif len(split_args_json) == 1:
                            tc_original["function"]["arguments"] = split_args_json[0]
                            expanded_tool_calls.append(tc_original)
                        elif not split_args_json and args_str.strip():
                            expanded_tool_calls.append(tc_original)
                
                assembled_tool_calls = expanded_tool_calls
                full_turn_content = "".join(current_turn_content_parts)
                final_content = full_turn_content

                if not assembled_tool_calls:
                    if full_turn_content: 
                        self.add(role="assistant", content=full_turn_content)
                    break

                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if full_turn_content: 
                    assistant_msg["content"] = full_turn_content
                assistant_msg["tool_calls"] = assembled_tool_calls
                self.add(**assistant_msg)

                # --- BUCLE DE EJECUCIÓN DE HERRAMIENTAS MODIFICADO ---
                for tc_to_run in assembled_tool_calls:
                    tool_name = tc_to_run["function"]["name"]
                    tool_call_id = tc_to_run["id"]
                    tool = self._tools.get(tool_name)
                    result: Union[str, Generator[StreamingEvent, None, None]]
                    
                    if not tool:
                        result = json.dumps({"error": f"La herramienta '{tool_name}' no está registrada."})
                    else:
                        try:
                            tool_args_val = tc_to_run["function"].get("arguments")
                            tool_args_str = tool_args_val if isinstance(tool_args_val, str) else json.dumps(tool_args_val or {})
                            if not tool_args_str.strip(): 
                                tool_args_str = "{}"
                            parsed_args = json.loads(tool_args_str)
                            
                            yield StreamingEvent(
                                event_type="TOOL_CALL",
                                source_agent=self.name,
                                data=ToolCallData(tool_name=tool_name, tool_args=parsed_args, tool_call_id=tool_call_id)
                            )
                            result = tool(**parsed_args)
                        except json.JSONDecodeError as exc:
                            logger.warning(f"Argumentos JSON inválidos para '{tool_name}': {exc}. Recibido: '{tool_args_str}'")
                            result = json.dumps({"error": f"Argumentos JSON inválidos para '{tool_name}': {exc}. Recibido: '{tool_args_str}'"})
                        except Exception as e:
                            logger.error(f"Error inesperado preparando herramienta '{tool_name}': {e}", exc_info=True)
                            result = json.dumps({"error": f"Error inesperado preparando herramienta '{tool_name}': {e}"})

                    # --- LÓGICA PARA MANEJAR RESULTADOS SIMPLES O STREAMS ANIDADOS ---
                    result_content_for_memory = ""
                    if hasattr(result, '__iter__') and not isinstance(result, str):
                        # El resultado es un generador de eventos (un worker anidado)
                        nested_event_generator = result
                        accumulated_chunks = []
                        for nested_event in nested_event_generator:
                            yield nested_event # Reenviar evento del worker anidado
                            if nested_event.event_type == "LLM_CHUNK":
                                accumulated_chunks.append(str(nested_event.data))
                        result_content_for_memory = "".join(accumulated_chunks)
                    else:
                        # El resultado es un string simple de una herramienta normal
                        result_content_for_memory = str(result)

                    yield StreamingEvent(
                        event_type="TOOL_RESULT",
                        source_agent=self.name,
                        data=ToolResultData(tool_name=tool_name, tool_result=result_content_for_memory, tool_call_id=tool_call_id)
                    )
                    self.add(role="tool", content=result_content_for_memory, tool_call_id=tool_call_id, name=tool_name)
            
            else: # Se alcanzó MAX_TOOL_ITER
                warn_msg = f"Agente '{self.name}' alcanzó el máximo de {self.MAX_TOOL_ITER} iteraciones de herramientas."
                logger.warning(warn_msg)
                yield StreamingEvent(event_type="AGENT_WARN", source_agent=self.name, data=WarnData(warn_message=warn_msg))

        except Exception as e:
            logger.error(f"Error irrecuperable en el agente '{self.name}': {e}", exc_info=True)
            yield StreamingEvent(event_type="AGENT_ERROR", source_agent=self.name, data=ErrorData(error_message=str(e)))
            return

        yield StreamingEvent(
            event_type="AGENT_FINISH",
            source_agent=self.name,
            data=AgentFinishData(final_message=final_content)
        )

    def respond(self, user_input: str) -> Generator[StreamingEvent, None, None]:
        """
        Método principal para interactuar con el agente.
        Siempre devuelve un generador de eventos de streaming (`StreamingEvent`).
        El flag `self.stream` ahora solo controla si el LLM se consume en modo streaming,
        pero la salida de este método es consistente.
        """
        return self._process_agent_logic(user_input)

    def tool_exists(self, name: str) -> bool:
        """Verifica si una herramienta está registrada."""
        return name in self._tools

    def unregister_tool(self, name: str) -> bool:
        """Elimina una herramienta. Devuelve True si se eliminó, False si no."""
        if name not in self._tools:
            return False
        self._tools.pop(name)
        self._tool_defs = [d for d in self._tool_defs if d["function"]["name"] != name]
        return True

    def register_tool(self, tool: Tool) -> None:
        """Registra una herramienta, reemplazando si ya existe una con el mismo nombre."""
        if self.tool_exists(
            tool.name
        ):  # Des-registrar para evitar duplicados en _tool_defs
            self.unregister_tool(tool.name)
        self._tools[tool.name] = tool
        self._tool_defs.append({"type": "function", "function": tool.schema})
