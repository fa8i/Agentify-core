from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

# Tipos de eventos que nuestro sistema puede emitir
EventType = Literal[
    "AGENT_START",
    "LLM_CHUNK",
    "TOOL_CALL",
    "TOOL_RESULT",
    "AGENT_FINISH",
    "AGENT_ERROR",
    "AGENT_WARN",
]


@dataclass(frozen=True)
class StreamingEvent:
    """
    Representa un único evento estructurado en el flujo de ejecución de un agente.
    Es inmutable (frozen=True) para garantizar la consistencia.
    """

    event_type: EventType
    source_agent: str
    data: Any = None

    def __repr__(self) -> str:
        return f"StreamingEvent(type={self.event_type}, agent='{self.source_agent}', data={self.data})"


# Estructuras de datos específicas para los eventos


@dataclass(frozen=True)
class ToolCallData:
    """Datos para un evento TOOL_CALL."""

    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str


@dataclass(frozen=True)
class ToolResultData:
    """Datos para un evento TOOL_RESULT."""

    tool_name: str
    tool_result: Any
    tool_call_id: str


@dataclass(frozen=True)
class AgentStartData:
    """Datos para un evento AGENT_START."""

    user_input: str


@dataclass(frozen=True)
class AgentFinishData:
    """Datos para un evento AGENT_FINISH."""

    final_message: Optional[str] = None


@dataclass(frozen=True)
class ErrorData:
    """Datos para un evento AGENT_ERROR."""

    error_message: str
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class WarnData:
    """Datos para un evento AGENT_WARN."""

    warn_message: str
