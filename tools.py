from base_agent import Tool
import datetime
import ast
import operator as op
import os
import requests
from dotenv import load_dotenv

load_dotenv()

get_current_time_schema = {
    "name": "get_current_time",
    "description": "Devuelve la hora y fecha actual en formato ISO 8601.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

def get_current_time():
    now = datetime.datetime.now().astimezone().isoformat()
    return {"current_time": now}


calculate_expression_schema = {
    "name": "calculate_expression",
    "description": "Evalúa una expresión matemática segura y devuelve el resultado.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Expresión matemática a calcular, por ejemplo '2 + 2 * (3 - 1)'.",
            }
        },
        "required": ["expression"],
    },
}

_allowed_ops = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

def _eval_node(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _allowed_ops[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        return _allowed_ops[type(node.op)](operand)
    raise ValueError(f"Operador no permitido: {node}")

def calculate_expression(expression: str):
    try:
        tree = ast.parse(expression, mode='eval').body
        result = _eval_node(tree)
        return {"result": result}
    except Exception as e:
        return {"error": f"Expresión inválida: {e}"}


get_weather_schema = {
    "name": "get_weather",
    "description": "Obtiene el estado del tiempo o clima actual para una ciudad o zona especificada.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Nombre de la ciudad o zona para consultar el clima.",
            }
        },
        "required": ["location"],
    },
}

def get_weather(location: str):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "Variable de entorno OPENWEATHER_API_KEY no configurada."}
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": api_key, "units": "metric"}
        )
        data = response.json()
        if response.status_code != 200:
            return {"error": data.get("message", "Error desconocido al obtener el clima.")}
        weather = {
            "location": data["name"],
            "description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return {"weather": weather}
    except Exception as e:
        return {"error": f"Error al conectar con el servicio de clima: {e}"}


get_current_time_tool = Tool(get_current_time_schema, get_current_time)
calculate_expression_tool = Tool(calculate_expression_schema, calculate_expression)
get_weather_tool = Tool(get_weather_schema, get_weather)

tools = [
    get_current_time_tool, 
    calculate_expression_tool, 
    get_weather_tool
    ]