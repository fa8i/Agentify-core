from typing import Generator
from agentify.events import StreamingEvent


class ConsoleRenderer:
    """
    Renderiza los StreamingEvents en la consola para una experiencia interactiva.
    Interpreta los eventos del agente y los formatea para ser legibles.
    """

    def __init__(self):
        self.current_agent = None

    def render(self, event_generator: Generator[StreamingEvent, None, None]) -> None:
        """Consume el generador de eventos y los imprime en la consola."""
        full_response_text = ""
        print("\n\033[1m[Sistema]:\033[0m ", end="", flush=True)

        for event in event_generator:
            # Si la fuente del evento cambia, mostramos qué agente está activo
            if self.current_agent != event.source_agent:
                self.current_agent = event.source_agent
                print(
                    f"\n\033[90m--- Agente Activo: {self.current_agent} ---\033[0m", # grey
                    flush=True,
                )
                # Al cambiar de agente, reiniciamos la línea de impresión
                if event.event_type != "AGENT_FINISH":
                    print("\n\033[1m[Sistema]:\033[0m ", end="", flush=True)

            if event.event_type == "LLM_CHUNK":
                text_chunk = event.data
                full_response_text += text_chunk
                print(text_chunk, end="", flush=True)

            elif event.event_type == "TOOL_CALL":
                tool_name = event.data.tool_name
                tool_args = event.data.tool_args
                print(
                    f"\n\n\033[93m[Llamando a la herramienta: `{tool_name}` con argumentos: {tool_args}]\033[0m", # yellow
                    flush=True,
                )

            elif event.event_type == "TOOL_RESULT":
                tool_name = event.data.tool_name
                result_preview = str(event.data.tool_result)[
                    :250
                ]
                print(
                    f"\n\033[92m[Resultado de `{tool_name}`: {result_preview}...]\033[0m", # green (preview)
                    flush=True,
                )

                print("\n\033[1m[Sistema]:\033[0m ", end="", flush=True)

            elif event.event_type == "AGENT_ERROR":
                print(
                    f"\n\n\033[91m[ERROR en {event.source_agent}]: {event.data.error_message}\033[0m", # red
                    flush=True,
                )

        print()
