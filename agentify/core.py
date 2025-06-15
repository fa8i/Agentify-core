from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from agentify.events import StreamingEvent


@dataclass(slots=True)
class Tool:
    """Wrapper de vinculación de JSON-schema con su función Python."""

    schema: Dict[str, Any]
    func: Callable[..., Any]

    def __post_init__(self) -> None:
        if "name" not in self.schema:
            raise ValueError("Tool schema must include 'name'")

    @property
    def name(self) -> str:
        return self.schema["name"]

    def __call__(self, **kwargs: Any) -> Union[str, Generator[StreamingEvent, None, None]]:
        try:
            # La función envuelta puede ser un generador
            result = self.func(**kwargs)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        
        # Si es un generador, lo devolvemos tal cual. Si no, lo convertimos a string.
        if hasattr(result, '__iter__') and not isinstance(result, str):
            return result
        
        return str(result)

class BaseMemory(ABC):
    """Interfaz abstracta para los sistemas de memoria de los agentes."""
    @abstractmethod
    def add(self, message: Dict[str, Any]) -> None: pass
    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]: pass
    @abstractmethod
    def clear(self) -> None: pass
    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None: pass


class BaseAgent(ABC):
    """Clase base abstracta para todos los agentes del framework."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        memory_type: str = "in_memory",
        memory_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Inicializa un agente.
        
        Args:
            name (str): Nombre único para el agente.
            description (str): Descripción de lo que hace el agente (crucial para el orquestador).
            system_prompt (str): El prompt del sistema que define el comportamiento del agente.
            memory_type (str): El tipo de memoria a utilizar (ej: 'in_memory').
            memory_config (Optional[Dict[str, Any]]): Configuración adicional para la memoria.
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.memory = self._create_memory(memory_type, system_prompt, memory_config)

    def _create_memory(self, memory_type: str, system_prompt: str, config: Optional[Dict[str, Any]]) -> BaseMemory:
        """Fábrica para crear instancias de memoria."""
        if memory_type == "in_memory":
            # Importación local para evitar dependencias circulares
            from agentify.memory import InMemoryMemory 
            return InMemoryMemory(system_prompt=system_prompt)
        else:
            raise ValueError(f"Tipo de memoria desconocido: {memory_type}")

    @abstractmethod
    def respond(self, user_input: str) -> Union[str, Generator[str, None, None]]:
        """Método principal para interactuar con el agente."""
        pass
    
    def clear_memory(self) -> None:
        """Limpia la memoria del agente."""
        self.memory.clear()