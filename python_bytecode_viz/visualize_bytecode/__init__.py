import os
import shutil
import json

from .process_bytecode import function_cfg_to_dict

from typing import Callable

_MODULE_DIR = os.path.abspath(os.path.dirname(__file__))

VISUALIZATION_HTML_LOCATION = os.path.join(
    _MODULE_DIR, "web_templates", "visualization.html"
)
VISUALIZATION_CSS_LOCATION = os.path.join(
    _MODULE_DIR, "web_templates", "visualization.css"
)
VISUALIZATION_JS_LOCATION = os.path.join(
    _MODULE_DIR, "web_templates", "visualization.js"
)

def visualize_bytecode(func: Callable, output_dir: str) -> None:
    json_dict = function_cfg_to_dict(func)

    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "bytecode.json"), "w") as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)

    shutil.copy(VISUALIZATION_HTML_LOCATION, output_dir)
    shutil.copy(VISUALIZATION_CSS_LOCATION, output_dir)
    shutil.copy(VISUALIZATION_JS_LOCATION, output_dir)
    
    return
