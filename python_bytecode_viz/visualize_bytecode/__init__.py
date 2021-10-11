
import json
from .process_bytecode import function_cfg_to_dict

from typing import List, Callable 

def visualize_bytecode(func: Callable) -> None:
    json_dict = function_cfg_to_dict(func)
    print(json.dumps(json_dict, sort_keys=True, indent=4))
    return
