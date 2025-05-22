import os
from dotenv import load_dotenv
from typing import Union, Dict, Any, Optional, Callable
from openai import OpenAI, AzureOpenAI

load_dotenv()

LLMClientType = Union[OpenAI, AzureOpenAI]

ClientBuilder = Callable[[Dict[str, Any], int], LLMClientType]

class LLMClientFactory:
    """Clase para crear clientes LLM para diferentes proveedores."""
    SUPPORTED_PROVIDERS = ["azure", "openai", "deepseek", "gemini"]

    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout
        self._builders: Dict[str, ClientBuilder] = {
            "openai": self._create_openai_client,
            "deepseek": self._create_deepseek_client,
            "gemini": self._create_gemini_client,
            "azure": self._create_azure_client,
        }

    def _get_env_or_config(self, key: str, env_var_name: str, config: Dict[str, Any], required: bool = True) -> Optional[str]:
        """Helper para obtener valor de config_override o variable de entorno."""
        value = config.get(key, os.getenv(env_var_name))
        if required and not value:
            raise ValueError(
                f"El parámetro '{key}' (o la variable de entorno '{env_var_name}') "
                f"es requerido pero no se encontró."
            )
        return value

    def _create_openai_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "OPENAI_API_KEY", config)
        return OpenAI(
            api_key=api_key,
            timeout=timeout,
        )

    def _create_deepseek_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        api_key = self._get_env_or_config("api_key", "DEEPSEEK_API_KEY", config)
        base_url = self._get_env_or_config("base_url", "DEEPSEEK_URL", config)
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _create_gemini_client(self, config: Dict[str, Any], timeout: int) -> OpenAI:
        # Utilización de Gemini mediante endpoint para el uso de funcrion calling de OpenAI
        api_key = self._get_env_or_config("api_key", "GEMINI_API_KEY", config)
        base_url = self._get_env_or_config("base_url", "GEMINI_URL", config, required=False)
        
        client_args = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_args["base_url"] = base_url
        else:
            print("Advertencia: No se proporcionó GEMINI_URL para Gemini. "
                  "Asumiendo que el SDK de OpenAI lo maneja o no es necesario.")
        
        return OpenAI(**client_args)

    def _create_azure_client(self, config: Dict[str, Any], timeout: int) -> AzureOpenAI:
        api_key = self._get_env_or_config("api_key", "AZURE_OPENAI_KEY", config)
        api_version = self._get_env_or_config("api_version", "API_VERSION", config)
        azure_endpoint = self._get_env_or_config("azure_endpoint", "AZURE_OPENAI_ENDPOINT", config)
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            timeout=timeout,
        )

    def create_client(
        self,
        provider: str,
        config_override: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> LLMClientType:
        """
        Crea un cliente LLM para el proveedor especificado.

        Args:
            provider: Nombre del proveedor (ej. "openai", "azure").
            config_override: Un diccionario para sobreescribir o proveer parámetros de configuración
                             (ej. api_key, base_url) en lugar de usar variables de entorno.
            timeout: Timeout específico para este cliente, si no se usa el default_timeout de la factoría.

        Returns:
            Una instancia del cliente LLM.

        Raises:
            ValueError: Si el proveedor no es soportado o faltan parámetros requeridos.
        """
        provider_lower = provider.lower()
        if provider_lower not in self._builders:
            raise ValueError(
                f"Proveedor '{provider}' no soportado. "
                f"Soportados: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        builder_func = self._builders[provider_lower]
        effective_timeout = timeout if timeout is not None else self.default_timeout
        effective_config = config_override or {}

        return builder_func(effective_config, effective_timeout)