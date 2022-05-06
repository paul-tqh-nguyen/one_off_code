
import ast
import argparse
import os

from typing import List

def check_func_def_ast_node(func_ast_node: ast.FunctionDef, python_file: str) -> None:
    func_name = func_ast_node.name
    for arg in func_ast_node.args.args:
        # currently doesn't handle args or kwargs
        if arg.annotation is None:
            arg_name = arg.arg
            print(f'{python_file}:{arg.lineno}:{arg.col_offset}: The callable {repr(func_name)} is missing a type hint for arg named {repr(arg_name)}.')
    if func_ast_node.returns is None:
        print(f'{python_file}:{func_ast_node.lineno}:{func_ast_node.col_offset}: The callable {repr(func_name)} is missing a return type hint.')

def find_python_files(search_directory: str) -> List[str]:
    python_files = []
    for root, _, files in os.walk(search_directory):
        for f in files:
            if f.endswith(".py"):
                f_abs_path = os.path.join(root, f)
                python_files.append(f_abs_path)
    return python_files

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-directory', default='.', help="Directory to search in. Defaults to the current working directory.")
    args = parser.parse_args()
    search_directory = os.path.abspath(args.search_directory)
    python_files = find_python_files(search_directory)
    for python_file in python_files:
        with open(python_file, 'r') as f:
            source = f.read()
        source_ast = ast.parse(source, filename=python_file)
        def_defined_callables = [e for e in ast.walk(source_ast) if isinstance(e, ast.FunctionDef)]
        for func_ast_node in def_defined_callables:
            check_func_def_ast_node(func_ast_node, python_file)

if __name__ == '__main__':
    main()
