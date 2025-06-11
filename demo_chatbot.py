import gradio as gr
from agentify.base_agent import BaseAgent
from agentify.client_builder import LLMClientFactory
from agentify.prompts import assistant_prompt
from agentify.tools import (
    get_current_time_tool,
    calculate_expression_tool,
    get_weather_tool,
)

BUILTIN_TOOLS = {
    "get_current_time": get_current_time_tool,
    "calculate_expression": calculate_expression_tool,
    "get_weather": get_weather_tool,
}

PROVIDERS = ["azure", "openai", "deepseek", "gemini", "anthropic"]

PROVIDER_MODELS = {
    "azure": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "gemini": ["gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-06-05", "gemini-2.0-flash"],
    "anthropic": ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    "llama": ["Llama-4-Scout-17B-16E-Instruct-FP8", "Cerebras-Llama-4-Scout-17B-16E-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8", "Groq-Llama-4-Maverick-17B-128E-Instruct", "Llama-3.3-70B-Instruct"]
}


def create_agent_instance(name_val, provider_val, model_val, temperature_val, timeout_val, stream_val, selected_tools_val):
    """Crea una nueva instancia del agente con los parámetros especificados."""
    tools_val = selected_tools_val or []
    tools = [BUILTIN_TOOLS[t] for t in tools_val if t in BUILTIN_TOOLS]
    return BaseAgent(
        name=name_val or "GradioAgent",
        system_prompt=assistant_prompt,
        provider=provider_val,
        model_name=model_val,
        temperature=temperature_val,
        tools=tools,
        agent_timeout=timeout_val,
        stream=stream_val,
        client_factory=LLMClientFactory(),
    )


def stream_response_to_chatbot(message, agent_instance, chat_history_list):
    """Maneja el streaming de respuestas para el chatbot."""
    chat_history_list = chat_history_list or []
    chat_history_list.append([message, ""])

    response_stream = agent_instance.respond(message)

    if isinstance(response_stream, str):
        chat_history_list[-1][1] = response_stream
        yield chat_history_list
    else:
        full_response = ""
        for chunk in response_stream:
            full_response += chunk
            chat_history_list[-1][1] = full_response
            yield chat_history_list


def clear_agent_memory(agent_instance):
    """Limpia la memoria del agente."""
    if agent_instance:
        agent_instance.clear_memory()


def update_model_dropdown(provider):
    """Actualiza las opciones del dropdown de modelos basado en el proveedor seleccionado."""
    models = PROVIDER_MODELS.get(provider, ["gpt-4.1-nano"])
    return gr.Dropdown(choices=models, value=models[0])


def build_interface():
    with gr.Blocks(
        fill_height=True,
        fill_width=True,
        title="Agentify Chatbot"
    ) as demo:
        
        # Estado del agente
        agent_state = gr.State()
        
        # Sidebar con configuración
        with gr.Sidebar():
            
            gr.Markdown("## Model Settings:")

            name_input = gr.Textbox(
                label="Agent Name",
                placeholder="Enter agent name",
                value="GradioAgent"
            )
            
            # Configuración del modelo
            provider_input = gr.Dropdown(
                label="Provider",
                choices=PROVIDERS,
                value="openai"
            )
            
            model_input = gr.Dropdown(
                label="Model",
                choices=PROVIDER_MODELS["openai"],
                value="gpt-4.1-nano"
            )
            
            temperature_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.7,
                step=0.05,
                label="Temperature"
            )
            
            gr.Markdown("## Tools & Advanced:")
            
            # Herramientas disponibles
            tools_checkbox_group = gr.CheckboxGroup(
                label="Available Tools",
                choices=list(BUILTIN_TOOLS.keys()),
                value=["get_current_time"]
            )
            
            # Configuración avanzada
            timeout_number = gr.Number(
                label="Timeout (seconds)",
                value=60,
                minimum=1,
                maximum=300
            )
            
            stream_checkbox = gr.Checkbox(
                label="Stream Responses",
                value=True
            )
            
            # Botones de control
            rebuild_button = gr.Button(
                "🔄 Create/Reset Agent",
                variant="primary"
            )
            
            clear_button = gr.Button(
                "🗑️ Clear Conversation",
                variant="secondary"
            )
            
            # Estado del agente
            gr.Markdown("## Agent Status:")
            agent_status = gr.Textbox(
                label="Status",
                value="Not Initialized",
                interactive=False
            )
            
            active_tools = gr.Textbox(
                label="Active Tools",
                value="None",
                interactive=False
            )
        
        # Área principal
        gr.Markdown("<div style='text-align: center;'><h1>Agentify Chatbot</h1></div>")        
        
        chatbot_display = gr.Chatbot(
            scale=1,
            label="Conversation",
            bubble_full_width=False,
            group_consecutive_messages=False,
            height=500
        )
        
        message_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here and press Enter...",
            submit_btn=True
        )

        # --- Event Handlers ---
        
        def agent_creation_logic(name_val, provider_val, model_val, temp_val, time_val, stream_val, tools_list_val):
            """Crea un agente y actualiza el estado."""
            try:
                agent = create_agent_instance(name_val, provider_val, model_val, temp_val, time_val, stream_val, tools_list_val)
                status = f"✅ Active - {provider_val}/{model_val}"
                tools_text = ", ".join(tools_list_val) if tools_list_val else "None"
                return agent, status, tools_text
            except Exception as e:
                return None, f"❌ Error: {str(e)}", "None"

        def handle_rebuild_click(name_val, provider_val, model_val, temp_val, time_val, stream_val, tools_list_val):
            """Maneja el clic del botón rebuild."""
            new_agent, status, tools_text = agent_creation_logic(name_val, provider_val, model_val, temp_val, time_val, stream_val, tools_list_val)
            return new_agent, [], "", status, tools_text

        def handle_send_message(user_msg, chat_hist, current_agent):
            """Maneja el envío de mensajes."""
            if not user_msg.strip():
                yield chat_hist or []
                return

            if current_agent is None:
                error_msg = "❌ Error: Agent not initialized. Please click 'Create/Reset Agent' first."
                updated_hist = (chat_hist or []) + [[user_msg, error_msg]]
                yield updated_hist
                return

            yield from stream_response_to_chatbot(user_msg, current_agent, chat_hist)

        def handle_clear_conversation(current_agent):
            """Maneja la limpieza de la conversación."""
            clear_agent_memory(current_agent)
            return [], ""

        def update_provider_change(provider):
            """Actualiza el modelo cuando cambia el proveedor."""
            return update_model_dropdown(provider)

        # --- Event Connections ---
        
        provider_input.change(
            update_provider_change,
            inputs=[provider_input],
            outputs=[model_input]
        )
        
        demo.load(
            agent_creation_logic,
            inputs=[name_input, provider_input, model_input, temperature_slider, timeout_number, stream_checkbox, tools_checkbox_group],
            outputs=[agent_state, agent_status, active_tools]
        )

        rebuild_button.click(
            handle_rebuild_click,
            inputs=[name_input, provider_input, model_input, temperature_slider, timeout_number, stream_checkbox, tools_checkbox_group],
            outputs=[agent_state, chatbot_display, message_input, agent_status, active_tools]
        )

        message_input.submit(
            handle_send_message,
            inputs=[message_input, chatbot_display, agent_state],
            outputs=[chatbot_display]
        ).then(lambda: "", outputs=[message_input])

        clear_button.click(
            handle_clear_conversation,
            inputs=[agent_state],
            outputs=[chatbot_display, message_input]
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(
        share=False,
        inbrowser=True,
        show_error=True
    )