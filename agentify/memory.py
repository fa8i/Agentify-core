from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseMemory(ABC):
    """
    Interfaz abstracta para los sistemas de memoria de los agentes.
    Define el contrato que todas las implementaciones de memoria deben seguir.
    """

    @abstractmethod
    def add(self, message: Dict[str, Any]) -> None:
        """Añade un único mensaje al historial de la memoria."""
        pass

    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """Devuelve una copia del historial completo de mensajes."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Limpia la memoria, eliminando todos los mensajes excepto el prompt del sistema inicial.
        """
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None:
        """Establece o actualiza el mensaje del sistema."""
        pass


class InMemoryMemory(BaseMemory):
    """
    Una implementación simple de memoria que guarda el historial completo en una lista en memoria.
    Ideal para prototipado rápido y conversaciones de longitud moderada.
    """

    def __init__(self, system_prompt: str):
        """
        Inicializa la memoria con un prompt del sistema.

        Args:
            system_prompt (str): El mensaje que define el rol y comportamiento del agente.
        """
        self._system_prompt_content: str = system_prompt
        self._history: List[Dict[str, Any]] = []
        self.set_system_prompt(system_prompt)

    def add(self, message: Dict[str, Any]) -> None:
        """
        Añade un mensaje al final del historial.

        Args:
            message (Dict[str, Any]): El mensaje a añadir, con formato OpenAI (ej: {'role': 'user', 'content': '...'})
        """
        self._history.append(message)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Devuelve una copia superficial del historial para evitar mutaciones externas.

        Returns:
            List[Dict[str, Any]]: Una copia de la lista del historial.
        """
        return list(self._history)

    def clear(self) -> None:
        """
        Reinicia el historial a su estado inicial, conteniendo únicamente el prompt del sistema.
        Esencial para iniciar nuevas conversaciones con la misma instancia de agente.
        """
        self._history.clear()
        self.set_system_prompt(self._system_prompt_content)

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Establece el mensaje del sistema. Si ya existe uno, lo reemplaza.
        Si no, lo añade al principio de la lista.
        """
        self._system_prompt_content = system_prompt
        system_message = {"role": "system", "content": self._system_prompt_content}

        if self._history and self._history[0].get("role") == "system":
            self._history[0] = system_message
        else:
            self._history.insert(0, system_message)
