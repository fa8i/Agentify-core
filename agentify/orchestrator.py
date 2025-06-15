import json
from typing import Any, Dict, Generator, List

from agentify.core import BaseAgent, Tool
from agentify.worker import WorkerAgent
from agentify.events import (
    StreamingEvent,
    AgentStartData,
    ToolCallData,
    AgentFinishData,
    ErrorData,
)


class OrchestratorAgent(WorkerAgent):
    """Agente 'Supervisor' que puede responder directamente a preguntas simples
    o delegar tareas complejas a un equipo de WorkerAgents.
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        workers: List[BaseAgent],
        **kwargs: Any,
    ):
        agent_tools = [self._create_tool_from_agent(worker) for worker in workers]

        kwargs_for_base = kwargs.copy()
        kwargs_for_base["stream"] = True

        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
            **kwargs_for_base,
        )

        self.workers = {worker.name: worker for worker in workers}

    def _create_tool_from_agent(self, agent: BaseAgent) -> Tool:
        schema = {
            "name": agent.name,
            "description": agent.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": f"La pregunta completa y original del usuario para el agente {agent.name}.",
                    }
                },
                "required": ["user_input"],
            },
        }

        def agent_responder_func(
            user_input: str,
        ) -> Generator[StreamingEvent, None, None]:
            return agent.respond(user_input)

        return Tool(schema=schema, func=agent_responder_func)

    def respond(self, user_input: str) -> Generator[StreamingEvent, None, None]:
        yield StreamingEvent(
            event_type="AGENT_START",
            source_agent=self.name,
            data=AgentStartData(user_input=user_input),
        )
        self.add(role="user", content=user_input)

        try:
            response_stream = self._completion()

            assembled_tool_calls = []
            tool_call_assembler: Dict[int, Dict[str, Any]] = {}
            has_text_response = False

            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    has_text_response = True
                    yield StreamingEvent(
                        event_type="LLM_CHUNK",
                        source_agent=self.name,
                        data=delta.content,
                    )

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_assembler:
                            tool_call_assembler[idx] = {
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        call_data = tool_call_assembler[idx]
                        if tc_delta.id and not call_data["id"]:
                            call_data["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                call_data["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                call_data["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )

            for idx in sorted(tool_call_assembler.keys()):
                call_data = tool_call_assembler[idx]
                if call_data.get("function", {}).get("name"):
                    assembled_tool_calls.append(call_data)

            if has_text_response and not assembled_tool_calls:
                yield StreamingEvent(
                    event_type="AGENT_FINISH",
                    source_agent=self.name,
                    data=AgentFinishData(final_message="Respuesta directa completada."),
                )
                return

            if assembled_tool_calls:
                toolmaster_calls = [
                    tc
                    for tc in assembled_tool_calls
                    if tc["function"]["name"] == "ToolMaster"
                ]
                if len(toolmaster_calls) > 1:
                    assembled_tool_calls = [
                        {
                            "id": "forced_single_call",
                            "type": "function",
                            "function": {
                                "name": "ToolMaster",
                                "arguments": json.dumps({"user_input": user_input}),
                            },
                        }
                    ]

                for tc_to_run in assembled_tool_calls:
                    tool_name = tc_to_run["function"]["name"]
                    tool = self._tools.get(tool_name)

                    if not tool:
                        yield StreamingEvent(
                            event_type="AGENT_ERROR",
                            source_agent=self.name,
                            data=ErrorData(
                                error_message=f"Intenté llamar a un worker inexistente: {tool_name}"
                            ),
                        )
                        continue

                    parsed_args = json.loads(tc_to_run["function"]["arguments"] or "{}")
                    yield StreamingEvent(
                        event_type="TOOL_CALL",
                        source_agent=self.name,
                        data=ToolCallData(
                            tool_name=tool_name,
                            tool_args=parsed_args,
                            tool_call_id=tc_to_run.get("id", "N/A"),
                        ),
                    )

                    worker_event_generator = tool(**parsed_args)
                    yield from worker_event_generator

        except Exception as e:
            yield StreamingEvent(
                event_type="AGENT_ERROR",
                source_agent=self.name,
                data=ErrorData(
                    error_message=f"Error fatal en el orquestador: {str(e)}"
                ),
            )

        yield StreamingEvent(
            event_type="AGENT_FINISH",
            source_agent=self.name,
            data=AgentFinishData(final_message="Delegación completada."),
        )
