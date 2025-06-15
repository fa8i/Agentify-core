from agentify.prompts import orchestrator_system_prompt
from agentify.worker import WorkerAgent
from agentify.orchestrator import OrchestratorAgent
from agentify.tools import tools
from agentify.ui import ConsoleRenderer 

# Agente 1: Experto en programación
code_agent = WorkerAgent(
    name="CodeMaster",
    description="Excelente para generar, explicar o depurar código en cualquier lenguaje de programación. Su fuerte es Python, Java, Javascript, etc.",
    system_prompt="Eres un programador experto. Respondes con explicaciones claras y código de alta calidad. Solo hablas de programación.",
    provider="openai",
    model_name="gpt-4.1-mini",
    tools=[] # Este worker tiene su propia herramienta
)

# Agente 2: Expertoen tools de busqueda de hora, clima y matematicas
tool_agent = WorkerAgent(
    name="ToolMaster",
    description="Indispensable para responder preguntas que requieren información en tiempo real, como el tiempo/clima actual, la hora, o realizar operaciones matemáticas. Usa sus herramientas para obtener respuestas precisas.",
    system_prompt="Eres un asistente que utiliza herramientas para obtener respuestas factuales. Proporciona la información obtenida de forma clara y concisa.",
    provider="openai",
    model_name="gpt-4.1-mini",
    tools=tools # Este worker tiene su propia herramienta
)

# Agente 3: Escritor creativo
writer_agent = WorkerAgent(
    name="CreativeWriter",
    description="El mejor agente para escribir poemas, historias cortas, guiones, letras de canciones o cualquier tipo de texto creativo y artístico.",
    system_prompt="Eres un poeta y escritor galardonado. Tus respuestas son siempre evocadoras y artísticas. No escribes código ni respondes preguntas factuales.",
    provider="gemini",
    model_name="gemini-2.0-flash",
)

# Crear el Agente Orquestador

main_orchestrator = OrchestratorAgent(
    name="MainOrchestrator",
    description="El agente principal que enruta las tareas.",
    system_prompt=orchestrator_system_prompt,
    workers=[code_agent, writer_agent, tool_agent],
    provider="openai",
    model_name="gpt-4o",
    temperature=0.1,
    stream = True
)

renderer = ConsoleRenderer()

print("Bienvenido al sistema de agentes. ¿En qué puedo ayudarte?")

def run_prompt_with_renderer(prompt: str):
    """
    Ejecuta un prompt y renderiza la salida de eventos en tiempo real.
    """
    print("=" * 60)
    print(f"\033[1m[Usuario]:\033[0m {prompt}")
    
    # El método 'respond' ahora devuelve un generador de eventos
    event_generator = main_orchestrator.respond(prompt)
    
    # El renderer se encarga de consumir el generador y mostrar la salida
    renderer.render(event_generator)
    print("=" * 60 + "\n")


# --- 4. Ejecutar Casos de Uso ---

run_prompt_with_renderer(
    "Por favor dime que tiempo hace en Madrid, en Budapest y en Estocolmo."
)

# Caso de uso 2: Tarea creativa
run_prompt_with_renderer(
    "Escribe un poema sobre el clima de esas ciudades."
)
