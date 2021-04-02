
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

from functools import reduce
from typing import Dict, Callable, Tuple, Union, Optional
import typing_extensions

from .parser import (
    ASTNode,
    UnaryOperationExpressionASTNode,
    BinaryOperationExpressionASTNode,
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
from .ast_node import EMPTY_TENSOR_TYPE_AST_NODE
from .misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

###################
# Exception Types #
###################

class TypeInferenceFailure(Exception):
    pass

class TypeInferenceConsistencyError(Exception):

    def __init__(self, ast_node: ASTNode, *inconsistent_types: Union[TensorTypeASTNode, FunctionDefinitionASTNode]) -> None:
        self.ast_node = ast_node
        self.inconsistent_types = inconsistent_types
        super().__init__(f'{ast_node} has the following inconsistent types: {inconsistent_types}')
        return

#################################################
# Type Inference Consistency Checking Utilities #
#################################################

def _assert_type_consistency(type_a: Union[TensorTypeASTNode, FunctionDefinitionASTNode], type_b: Union[TensorTypeASTNode, FunctionDefinitionASTNode]) -> bool:
    if type_a == type_b:
        return True
    if type(type_a) == TensorTypeASTNode == type(type_b):
        if None in (type_a.shape, type_b.shape):
            return True
        if len(type_a.shape) == len(type_b.shape):
            if all(dim_a == dim_b for dim_a, dim_b in zip(type_a.shape, type_b.shape) if None not in (dim_a, dim_b)):
                return  True
    return False

def assert_type_consistency(ast_node: ASTNode, *types: Union[TensorTypeASTNode, FunctionDefinitionASTNode]) -> None:
    if not reduce(_assert_type_consistency, types):
        raise TypeInferenceConsistencyError(ast_node, *types)
    return

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
        def decorated_func(ast_node: ASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
            assert isinstance(ast_node, ast_node_type)
            var_name_to_type_info  = dict(var_name_to_type_info)
            return type_inference_method(ast_node, var_name_to_type_info)
        AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD[ast_node_type] = decorated_func
        return decorated_func
    return decorator

##########################################
# Expression Node Type Inference Methods #
##########################################

def determine_expression_ast_node_type(ast_node: ExpressionASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> TensorTypeASTNode:
    inferred_type = BOGUS_TOKEN
    if isinstance(ast_node, NothingTypeLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='NothingType', shape=[])
    elif isinstance(ast_node, IntegerLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='Integer', shape=[])
    elif isinstance(ast_node, BooleanLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='Boolean', shape=[])
    elif isinstance(ast_node, FloatLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='Float', shape=[])
    elif isinstance(ast_node, FunctionCallExpressionASTNode):
        ASSERT.TypeInferenceFailure(ast_node.function_name in var_name_to_type_info, f'Return type of {ast_node.function_name} is not declared.')
        return_types = var_name_to_type_info[ast_node.function_name].function_return_types
        assert len(return_types) != 0
        if len(return_types) != 1:
            raise TypeInferenceConsistencyError(ast_node, *return_types)
        inferred_type = only_one(return_types)
    elif isinstance(ast_node, VariableASTNode):
        inferred_type = var_name_to_type_info[ast_node.name]
    elif isinstance(ast_node, UnaryOperationExpressionASTNode):
        inferred_type = determine_expression_ast_node_type(ast_node.arg, var_name_to_type_info)
    elif isinstance(ast_node, BinaryOperationExpressionASTNode):
        left_arg_inferred_type = determine_expression_ast_node_type(ast_node.left_arg, var_name_to_type_info)
        right_arg_inferred_type = determine_expression_ast_node_type(ast_node.right_arg, var_name_to_type_info)
        if left_arg_inferred_type == right_arg_inferred_type:
            inferred_type = left_arg_inferred_type
    elif isinstance(ast_node, VectorExpressionASTNode):
        element_types = (determine_expression_ast_node_type(value, var_name_to_type_info) for value in ast_node.values)
        element_types = quadratic_unique(element_types)
        if len(element_types) == 1:
            element_type = only_one(element_types)
            inferred_type = TensorTypeASTNode(base_type_name=element_type.base_type_name, shape=[len(ast_node.values)]+element_type.shape)
        elif len(element_types) == 0:
            raise NotImplementedError # TODO handle this case
        else:
            assert_type_consistency(ast_node, *element_types)
    if inferred_type is BOGUS_TOKEN:
        raise NotImplementedError(f'Could not determine type of {ast_node}.')
    return inferred_type

#####################################
# Registered Type Inference Methods #
#####################################

@register_type_inference_method(AssignmentASTNode)
def assignment_type_inference(ast_node: AssignmentASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    changed = False
    for variable_node, tensor_type_node in ast_node.variable_type_pairs:
        if tensor_type_node != EMPTY_TENSOR_TYPE_AST_NODE:
            if variable_node.name in var_name_to_type_info:
                assert_type_consistency(variable_node, var_name_to_type_info[variable_node.name], tensor_type_node)
            else:
                var_name_to_type_info[variable_node.name] = tensor_type_node
    if isinstance(ast_node.value, FunctionCallExpressionASTNode):
        ASSERT.TypeInferenceFailure(ast_node.value.function_name in var_name_to_type_info, f'Return type of {ast_node.value.function_name} is not declared.')
        return_types = var_name_to_type_info[ast_node.value.function_name].function_return_types
        if len(return_types) != len(ast_node.variable_type_pairs):
            raise TypeInferenceFailure(f'{ast_node} attempts to assign the result of {ast_node.value.function_name}, which returns {len(return_types)} values, to {len(ast_node.variable_type_pairs)} variables.')
        for return_index, return_type in enumerate(return_types):
            inferred_type = return_type
            variable_node, tensor_type_node = ast_node.variable_type_pairs[return_index]
            if tensor_type_node == EMPTY_TENSOR_TYPE_AST_NODE:
                ast_node.variable_type_pairs[return_index] = (variable_node, inferred_type)
                var_name_to_type_info[variable_node.name] = inferred_type
                changed = True
            else:
                assert_type_consistency(variable_node, inferred_type, var_name_to_type_info[variable_node.name])
    elif isinstance(ast_node.value, ExpressionASTNode):
       variable_node, tensor_type_node = only_one(ast_node.variable_type_pairs)
       inferred_type = determine_expression_ast_node_type(ast_node.value, var_name_to_type_info)
       if variable_node.name in var_name_to_type_info:
           assert_type_consistency(variable_node, inferred_type, var_name_to_type_info[variable_node.name])
       else:
           assert tensor_type_node == EMPTY_TENSOR_TYPE_AST_NODE
           ast_node.variable_type_pairs = [(variable_node, inferred_type)]
           var_name_to_type_info[variable_node.name] = inferred_type
           changed = True
    else:
        assert isinstance(ast_node.value, ASTNode)
        raise NotImplementedError(f'Type inference on node type {type(ast_node.value)} not yet supported.')
    return var_name_to_type_info, changed

@register_type_inference_method(ModuleASTNode)
def module_type_inference(ast_node: ModuleASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    changed = False
    for statement in ast_node.statements:
        var_name_to_type_info, statement_changed = perform_type_inference(statement, var_name_to_type_info)
        changed |= statement_changed
    return var_name_to_type_info, changed

###############
# Entry Point #
###############

def perform_type_inference(ast_node: ASTNode, var_name_to_type_info: Optional[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]] = None) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    '''Modifies the AST node in place. Runs until quiescence.'''
    if var_name_to_type_info is None:
        var_name_to_type_info = dict()
    ast_node_type = type(ast_node)
    type_inference_method = AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD[ast_node_type]
    var_name_to_type_info, changed = type_inference_method(ast_node, var_name_to_type_info)
    while changed:
        var_name_to_type_info, changed = type_inference_method(ast_node, var_name_to_type_info)
    return var_name_to_type_info, changed

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")
