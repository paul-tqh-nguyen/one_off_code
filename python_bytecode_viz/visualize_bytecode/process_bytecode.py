import dis
import io
import inspect
import networkx as nx

from typing import List, Callable 


def instruction_pretty_string(
    instruction: dis.Instruction, offset_width: int = 4
) -> str:
    fields = []
    fields.append(repr(instruction.offset).rjust(offset_width))
    fields.append(instruction.opname.ljust(dis._OPNAME_WIDTH))
    if instruction.arg is not None:
        fields.append(repr(instruction.arg).rjust(dis._OPARG_WIDTH))
        if instruction.argrepr:
            fields.append("(" + instruction.argrepr + ")")
    return " ".join(fields).rstrip()


NON_BRANCH_OP_NAMES = {
    "LOAD_CONST",
    "LOAD_FAST",
    "STORE_FAST",
    "COMPARE_OP",
    "INPLACE_ADD",
    "INPLACE_SUBTRACT",
    "RETURN_VALUE",
    "BINARY_MODULO",
    "POP_BLOCK",
    "SETUP_LOOP",
}


def function_cfg_to_dict(func: Callable) -> dict:
    """Convert the instructions of the func into a JSON-friendly dict."""

    # generate the graph
    graph = nx.DiGraph()
    previous_node_id = None
    for instruction in dis.get_instructions(func):
        node_id = instruction.offset
        assert node_id is not None
        graph.add_node(node_id)
        node_attributes = graph.nodes[node_id]
        node_attributes["pretty_strings"] = [instruction_pretty_string(instruction)]
        node_attributes["source_code_line_numbers"] = (
            [] if instruction.starts_line is None else [instruction.starts_line]
        )
        if previous_node_id is not None:
            graph.add_edge(previous_node_id, node_id)

        op_name = instruction.opname
        if op_name in NON_BRANCH_OP_NAMES:
            pass
        elif op_name in ("POP_JUMP_IF_FALSE", "JUMP_ABSOLUTE"):
            graph.add_edge(node_id, instruction.arg)
        else:
            # TODO handle JUMP_FORWARD
            raise NotImplementedError(
                f"Bytecode instruction {repr(op_name)} not yet supported"
            )

        previous_node_id = node_id

    assert nx.is_weakly_connected(graph)

    # compress the graph
    first_node_id = next(dis.get_instructions(func)).offset
    node_ids_to_visit = [first_node_id]
    while len(node_ids_to_visit) > 0:
        node_id = node_ids_to_visit.pop()
        neighbors = list(graph.neighbors(node_id))
        if len(neighbors) == 1:
            [neighbor_id] = neighbors
            neighbor_predecessors = list(graph.predecessors(neighbor_id))
            if len(neighbor_predecessors) == 1:
                # revisit since will have new neighbors
                node_ids_to_visit.append(node_id)
                assert neighbor_predecessors == [node_id]
                # combine the nodes
                for string in graph.nodes[neighbor_id]["pretty_strings"]:
                    graph.nodes[node_id]["pretty_strings"].append(string)
                for line_number in graph.nodes[neighbor_id]["source_code_line_numbers"]:
                    graph.nodes[node_id]["source_code_line_numbers"].append(line_number)
                for new_neighbor_id in graph.neighbors(neighbor_id):
                    graph.add_edge(node_id, new_neighbor_id)
                    # search the new neighbors
                    node_ids_to_visit.append(new_neighbor_id)
                graph.remove_node(neighbor_id)
        else:
            node_ids_to_visit += neighbors

    ans = nx.node_link_data(graph)

    string_io = io.StringIO()
    dis.dis(func, file=string_io)
    ans["bytecode_source_lines"] = string_io.getvalue().splitlines()

    lines, line_number = inspect.getsourcelines(func)
    ans["source_code_lines"] = lines
    ans["source_code_line_number"] = line_number
    
    return ans
