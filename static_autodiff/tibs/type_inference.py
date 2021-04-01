
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

from typing import Dict, Callable, Tuple, Union, Optional
import typing_extensions

from .parser import (
    ASTNode,
    PrintStatementASTNode,
    ComparisonExpressionASTNode,
    ExpressionASTNode,
    TensorTypeASTNode,
    VectorExpressionASTNode,
    FunctionDefinitionASTNode,
    ForLoopASTNode,
    WhileLoopASTNode,
    ConditionalASTNode,
    ScopedStatementSequenceASTNode,
    ReturnStatementASTNode,
    BooleanLiteralASTNode,
    IntegerLiteralASTNode,
    FloatLiteralASTNode,
    StringLiteralASTNode,
    NothingTypeLiteralASTNode,
    VariableASTNode,
    NegativeExpressionASTNode,
    ExponentExpressionASTNode,
    MultiplicationExpressionASTNode,
    DivisionExpressionASTNode,
    AdditionExpressionASTNode,
    SubtractionExpressionASTNode,
    NotExpressionASTNode,
    AndExpressionASTNode,
    XorExpressionASTNode,
    OrExpressionASTNode,
    GreaterThanExpressionASTNode,
    GreaterThanOrEqualToExpressionASTNode,
    LessThanExpressionASTNode,
    LessThanOrEqualToExpressionASTNode,
    EqualToExpressionASTNode,
    NotEqualToExpressionASTNode,
    StringConcatenationExpressionASTNode,
    FunctionCallExpressionASTNode,
    AssignmentASTNode,
    ModuleASTNode,
)
from .misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

#################################################
# Type Inference Consistency Checking Utilities #
#################################################

class TypeInferenceConsistencyError(Exception):

    def __init__(self, ast_node: ASTNode, *inconsistent_types: Union[TensorTypeASTNode, FunctionDefinitionASTNode]) -> None:
        self.ast_node = ast_node
        self.inconsistent_types = inconsistent_types
        super().__init__(f'{ast_node} has the following inconsistent types: {inconsistent_types}')
        return

def assert_type_consistency(ast_node: ASTNode, type_a: Union[TensorTypeASTNode, FunctionDefinitionASTNode], type_b: Union[TensorTypeASTNode, FunctionDefinitionASTNode]) -> None:
    if not type_a == type_b:
        raise TypeInferenceConsistencyError(ast_node, type_a, type_b)
    return
    
# def var_name_to_type_dicts_are_consistent(dict_a: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], dict_b: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> bool:
#     for key, dict_a_value in dict_a.keys():
#         if key in dict_b:
#             if dict_a_value != dict_b[key]:
#                 return False
#     return True

######################################
# Type Inference Method Registration #
######################################

AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD: Dict= {}

def register_type_inference_method(ast_node_type: type) -> Callable[
        [Callable[[ASTNode, Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]], Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]]],
        Callable[[ASTNode, Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]], Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]]
]:
    assert issubclass(ast_node_type, ASTNode)
    def decorator(
            type_inference_method: Callable[[ASTNode, Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]], Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]]
    ) -> Callable[[ASTNode, Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]], Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]]:
        '''
        type_inference_method can safely modify the given dict as this decorator automatically makes a copy before passing it in.
        The dict returned by type_inference_method should be a dict that can safely be reused by the parent AST node for performing more type inference.
        type_inference_method is expected to also do sanity checking that there are no type conflicts.
        '''
        assert ast_node_type not in AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD, f'Type inference method for {ast_node_type} redundantly declared.'
        def decorated_func(ast_node: ASTNode, var_name_to_type: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
            assert isinstance(ast_node, ast_node_type)
            var_name_to_type  = dict(var_name_to_type)
            return type_inference_method(ast_node, var_name_to_type)
        AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD[ast_node_type] = decorated_func
        return decorated_func
    return decorator

##########################
# Type Inference Methods #
##########################

def determine_expression_ast_node_type(ast_node: ExpressionASTNode) -> TensorTypeASTNode:
    inferred_type = BOGUS_TOKEN
    if isinstance(ast_node, NothingTypeLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='NothingType', shape=[])
    else:
        raise NotImplementedError(f'Could not determine type of {ast_node}.')
    assert inferred_type is not BOGUS_TOKEN
    return inferred_type

@register_type_inference_method(AssignmentASTNode)
def assignment_type_inference(ast_node: AssignmentASTNode, var_name_to_type: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    changed = False
    for variable_node, tensor_type_node in ast_node.variable_type_pairs:
        if tensor_type_node.base_type_name is not None:
            if variable_node.name in var_name_to_type:
                assert_type_consistency(variable_node, var_name_to_type[variable_node.name], tensor_type_node)
            else:
                var_name_to_type[variable_node.name] = tensor_type_node.base_type_name
    if isinstance(ast_node.value, FunctionCallExpressionASTNode):
        raise NotImplementedError # TODO support this
    elif isinstance(ast_node.value, ExpressionASTNode):
       variable_node, tensor_type_node = only_one(ast_node.variable_type_pairs)
       inferred_type = determine_expression_ast_node_type(ast_node.value)
       if variable_node.name in var_name_to_type:
           assert_type_consistency(variable_node, inferred_type, var_name_to_type[variable_node.name])
       else:
           assert tensor_type_node.base_type_name is None
           ast_node.variable_type_pairs = [(variable_node, inferred_type)]
    else:
        assert isinstance(ast_node.value, ASTNode)
        raise NotImplementedError(f'Type inference on node type {type(ast_node.value)} not yet supported.')
    return var_name_to_type, changed

@register_type_inference_method(ModuleASTNode)
def module_type_inference(ast_node: ModuleASTNode, var_name_to_type: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    changed = False
    for statement in ast_node.statements:
        var_name_to_type, statement_changed = perform_type_inference(statement, var_name_to_type)
        changed |= statement_changed
    return var_name_to_type, changed

###############
# Entry Point #
###############

def perform_type_inference(ast_node: ASTNode, var_name_to_type: Optional[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]] = None) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    '''Modifies the AST node in place. Runs until quiescence.'''
    if var_name_to_type is None:
        var_name_to_type = dict()
    ast_node_type = type(ast_node)
    type_inference_method = AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD[ast_node_type]
    var_name_to_type, changed = type_inference_method(ast_node, var_name_to_type)
    while changed:
        var_name_to_type, changed = type_inference_method(ast_node, var_name_to_type)
    return var_name_to_type, changed

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")
