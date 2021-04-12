
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import itertools
from functools import reduce
from typing import Dict, Callable, Tuple, Union, Optional
import typing_extensions

from .parser_utilities import SemanticError
from .parser import (
    ASTNode,
    UnaryOperationExpressionASTNode,
    BinaryOperationExpressionASTNode,
    StatementASTNode, 
    PrintStatementASTNode,
    ComparisonExpressionASTNode,
    BooleanExpressionASTNode,
    ExpressionASTNode,
    ArithmeticExpressionASTNode,
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

class TypeInferenceFailure(SemanticError):
    pass

class TypeInferenceConsistencyError(TypeInferenceFailure):

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
    if type(type_a) == TensorTypeASTNode == type(type_b) and type_a.base_type_name == type_b.base_type_name:
        if None in (type_a.shape, type_b.shape):
            return True
        if len(type_a.shape) == len(type_b.shape):
            if all(dim_a == dim_b for dim_a, dim_b in zip(type_a.shape, type_b.shape) if None not in (dim_a, dim_b)):
                return  True
    return False

def assert_type_consistency(ast_node: ASTNode, *types: Union[TensorTypeASTNode, FunctionDefinitionASTNode]) -> None:
    for type_1, type_2 in zip(types[1:], types):
        if not _assert_type_consistency(type_1, type_2):
            raise TypeInferenceConsistencyError(ast_node, type_1, type_2)
    return

######################################
# Type Inference Method Registration #
######################################

class TypeInferenceMethod:

    def __init__(
            self,
            ast_node_type: type,
            type_inference_method: Callable[[ASTNode, Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], Optional[str]], Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]]
    ) -> None:
        self.ast_node_type = ast_node_type
        self.type_inference_method = type_inference_method
        return

    def __call__(
            self,
            ast_node: ASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]],
            latest_function_name: Optional[str]
    ) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
        assert isinstance(ast_node, self.ast_node_type)
        var_name_to_type_info = dict(var_name_to_type_info)
        return self.type_inference_method(ast_node, var_name_to_type_info, latest_function_name)

AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD: Dict[type, TypeInferenceMethod] = {}

def register_type_inference_method(*ast_node_types: type, register_child_classes: bool = True) -> Callable[
        [Callable[[ASTNode, Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], Optional[str]], Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]]], 
        None
]:
    '''The parameter register_child_classes dictates if this handles ast_node_type as well as all child classes.'''
    def decorator(
            type_inference_method: Callable[[ASTNode, Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], Optional[str]], Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]]
    ) -> None:
        '''
        type_inference_method can safely modify the given dict as this decorator automatically makes a copy before passing it in.
        The dict returned by type_inference_method should be a dict that can safely be reused by the parent AST node for performing more type inference.
        type_inference_method is expected to also do sanity checking that there are no type conflicts.
        '''
        for ast_node_type in ast_node_types:
            child_ast_node_types = set(child_classes(ast_node_type)) if register_child_classes else ()
            child_ast_node_types = itertools.chain(child_ast_node_types, [ast_node_type])
            for child_ast_node_type in child_ast_node_types:
                assert issubclass(child_ast_node_type, ASTNode)
                assert child_ast_node_type not in AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD, f'Type inference method for {child_ast_node_type} redundantly declared.'
                AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD[child_ast_node_type] = TypeInferenceMethod(child_ast_node_type, type_inference_method)
        return
    return decorator

##########################################
# Expression Node Type Inference Methods #
##########################################

def determine_function_call_ast_node_type(
        ast_node: FunctionCallExpressionASTNode,
        var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]],
        check_return_value: bool
) -> Optional[TensorTypeASTNode]:
    ASSERT.TypeInferenceFailure(ast_node.function_name in var_name_to_type_info, f'Return type of {ast_node.function_name} is not declared.')
    function = var_name_to_type_info[ast_node.function_name]
    return_types = function.function_return_types
    assert len(return_types) != 0
    ASSERT.TypeInferenceFailure(len(ast_node.arg_bindings) == len(function.function_signature), f'{ast_node.function_name} given {len(ast_node.arg_bindings)} args when {len(function.function_signature)} were expected.')
    function_signature = {variable.name: variable_type for variable, variable_type in function.function_signature}
    ast_node_arg_bindings = {variable.name: variable_value for variable, variable_value in ast_node.arg_bindings}
    assert len(function_signature) == len(function.function_signature)
    assert len(ast_node_arg_bindings) == len(ast_node.arg_bindings)
    for variable_name, value in ast_node_arg_bindings.items():
        # TODO we can infer the types of sub-expressions of value that are variable AST nodes here
        value_tensor_type = determine_expression_ast_node_type(value, var_name_to_type_info)
        ASSERT.TypeInferenceFailure(variable_name in function_signature, f'{ast_node.function_name} given unexpected paramter binding for {repr(variable_name)}.')
        assert_type_consistency(ast_node, function_signature[variable_name], value_tensor_type)
    if check_return_value:
        if len(return_types) != 1:
            raise TypeInferenceFailure(f'{ast_node} returns multiple values ; cannot infer single type for this expression.')
        inferred_type = only_one(return_types)
    else:
        inferred_type = None
    return inferred_type

def determine_expression_ast_node_type(ast_node: ExpressionASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]) -> TensorTypeASTNode:
    '''Determine the type of an expression that's intended to be used inline. Function call AST nodes that return multiple values will cause an exception to be raised.'''
    inferred_type = BOGUS_TOKEN
    if isinstance(ast_node, NothingTypeLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='NothingType', shape=[])
    elif isinstance(ast_node, IntegerLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='Integer', shape=[])
    elif isinstance(ast_node, BooleanLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='Boolean', shape=[])
    elif isinstance(ast_node, FloatLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='Float', shape=[])
    elif isinstance(ast_node, StringLiteralASTNode):
        inferred_type = TensorTypeASTNode(base_type_name='String', shape=[])
    elif isinstance(ast_node, FunctionCallExpressionASTNode):
        inferred_type = determine_function_call_ast_node_type(ast_node, var_name_to_type_info, True)
    elif isinstance(ast_node, VariableASTNode):
        ASSERT.TypeInferenceFailure(ast_node.name in var_name_to_type_info, f'{ast_node.name} used before type declared.')
        inferred_type = var_name_to_type_info[ast_node.name]
    elif isinstance(ast_node, UnaryOperationExpressionASTNode) and isinstance(ast_node, ComparisonExpressionASTNode):
        arg_inferred_type = determine_expression_ast_node_type(ast_node.arg, var_name_to_type_info)
        inferred_type = TensorTypeASTNode(base_type_name='Boolean', shape=arg_inferred_type.shape)
    elif isinstance(ast_node, BinaryOperationExpressionASTNode) and isinstance(ast_node, ComparisonExpressionASTNode):
        left_arg_inferred_type = determine_expression_ast_node_type(ast_node.left_arg, var_name_to_type_info)
        right_arg_inferred_type = determine_expression_ast_node_type(ast_node.right_arg, var_name_to_type_info)
        ASSERT.TypeInferenceFailure(left_arg_inferred_type == right_arg_inferred_type, f'{ast_node} compares expressions with different types.')
        inferred_type = TensorTypeASTNode(base_type_name='Boolean', shape=left_arg_inferred_type.shape)
    elif isinstance(ast_node, UnaryOperationExpressionASTNode):
        assert isinstance(ast_node, (BooleanExpressionASTNode, ArithmeticExpressionASTNode, StringConcatenationExpressionASTNode))
        inferred_type = determine_expression_ast_node_type(ast_node.arg, var_name_to_type_info)
    elif isinstance(ast_node, BinaryOperationExpressionASTNode):
        assert isinstance(ast_node, (BooleanExpressionASTNode, ArithmeticExpressionASTNode, StringConcatenationExpressionASTNode))
        left_arg_inferred_type = determine_expression_ast_node_type(ast_node.left_arg, var_name_to_type_info)
        right_arg_inferred_type = determine_expression_ast_node_type(ast_node.right_arg, var_name_to_type_info)
        ASSERT.TypeInferenceFailure(left_arg_inferred_type == right_arg_inferred_type, f'{ast_node} operates on expressions with different types.')
        inferred_type = left_arg_inferred_type
    elif isinstance(ast_node, VectorExpressionASTNode):
        element_types = quadratic_unique(determine_expression_ast_node_type(value, var_name_to_type_info) for value in ast_node.values)
        ASSERT.TypeInferenceFailure(all(isinstance(element_type, TensorTypeASTNode) for element_type in element_types), f'{ast_node} must strictly contain tensor expressions.')
        ASSERT.TypeInferenceFailure(len({element_type.base_type_name for element_type in element_types}) == 1, f'{ast_node} must strictly contain tensor expressions with the same base type.')
        if len(element_types) == 1:
            element_type = only_one(element_types)
            inferred_type = TensorTypeASTNode(base_type_name=element_type.base_type_name, shape=[len(ast_node.values)]+element_type.shape)
        elif len(element_types) == 0:
            raise NotImplementedError # TODO handle this case
        else:
            # Pick the least restrictive shape
            assert_type_consistency(ast_node, *element_types)
            shapes = [element_type.shape for element_type in element_types]
            if None in shapes:
                inferred_type = TensorTypeASTNode(base_type_name=element_type[0].base_type_name, shape=None)
            else:
                assert len(set(map(len, shapes))) == 1
                shape = [None if None in dims else only_one(dims) for dims in zip(shapes)]
                inferred_type = TensorTypeASTNode(base_type_name=element_type[0].base_type_name, shape=None)
    else:
        raise NotImplementedError(f'Could not determine type of {ast_node}.')
    assert inferred_type is not BOGUS_TOKEN
    return inferred_type

#####################################
# Registered Type Inference Methods #
#####################################

@register_type_inference_method(ExpressionASTNode)
def expression_type_inference(ast_node: ASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    '''This simply verifies type consistency without inferring any new types.'''
    # TODO we can infer the types of sub-expressions that are variables here
    if isinstance(ast_node, FunctionCallExpressionASTNode):
        determine_function_call_ast_node_type(ast_node, var_name_to_type_info, False)
    else:
        determine_expression_ast_node_type(ast_node, var_name_to_type_info)
    return var_name_to_type_info, False

@register_type_inference_method(StatementASTNode, TensorTypeASTNode, register_child_classes=False)
def bogus_type_inference(ast_node: ASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    del ast_node
    del var_name_to_type_info
    del latest_function_name
    raise NotImplemented(f'Type inference is intentionally not implemented for instances of {StatementASTNode.__name__} or {TensorTypeASTNode.__name__}.')

@register_type_inference_method(PrintStatementASTNode)
def print_statement_type_inference(ast_node: PrintStatementASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    for value_to_print in ast_node.values_to_print:
        if isinstance(value_to_print, ExpressionASTNode):
            inferred_type = determine_expression_ast_node_type(value_to_print, var_name_to_type_info)
            ASSERT.TypeInferenceFailure(inferred_type.shape == [], f'{ast_node} attempts to print {value_to_print}, which is not 0-dimensional.')
    return var_name_to_type_info, False

@register_type_inference_method(FunctionDefinitionASTNode)
def function_definition_type_inference(ast_node: FunctionDefinitionASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    ASSERT.TypeInferenceFailure(ast_node.function_name not in var_name_to_type_info, f'{ast_node.function_name} defined multiple times.')
    ast_node.function_signature
    var_name_to_type_info[ast_node.function_name] = ast_node
    function_body_var_name_to_type_info = dict(var_name_to_type_info)
    redefined_variables = []
    for variable, variable_type in ast_node.function_signature:
        if variable.name in var_name_to_type_info:
            redefined_variables.add(variable.name)
        function_body_var_name_to_type_info[variable.name] = variable_type
    updated_var_name_to_type_info, changed = perform_type_inference(ast_node.function_body, function_body_var_name_to_type_info, ast_node.function_name)
    return var_name_to_type_info, changed

@register_type_inference_method(ReturnStatementASTNode)
def return_type_inference(ast_node: ReturnStatementASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    ASSERT.TypeInferenceFailure(latest_function_name is not None, f'Return statement used outside of function body.')
    ASSERT.TypeInferenceFailure(latest_function_name in var_name_to_type_info, f'Return type of {latest_function_name} is not declared.')
    expected_return_types = var_name_to_type_info[latest_function_name].function_return_types
    ASSERT.TypeInferenceFailure(len(expected_return_types) == len(ast_node.return_values), f'{latest_function_name} is declared to have {len(expected_return_types)} return values but attempts to return {len(ast_node.return_values)} values.')
    for expected_return_type, return_value in zip(expected_return_types, ast_node.return_values):
        return_type = determine_expression_ast_node_type(return_value, var_name_to_type_info)
        assert_type_consistency(ast_node, expected_return_type, return_type)
    return var_name_to_type_info, False

@register_type_inference_method(AssignmentASTNode)
def assignment_type_inference(ast_node: AssignmentASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    del latest_function_name
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
        if tensor_type_node != EMPTY_TENSOR_TYPE_AST_NODE:
            assert_type_consistency(variable_node, inferred_type, tensor_type_node)
        else:
            ast_node.variable_type_pairs = [(variable_node, inferred_type)]
            var_name_to_type_info[variable_node.name] = inferred_type
            changed = True
    else:
        assert isinstance(ast_node.value, ASTNode)
        raise NotImplementedError(f'Type inference on node type {type(ast_node.value)} not yet supported.')
    return var_name_to_type_info, changed

@register_type_inference_method(ModuleASTNode)
def module_type_inference(ast_node: ModuleASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    changed = False
    for statement in ast_node.statements:
        var_name_to_type_info, statement_changed = perform_type_inference(statement, var_name_to_type_info, latest_function_name)
        changed |= statement_changed
    return var_name_to_type_info, changed

@register_type_inference_method(ConditionalASTNode)
def conditional_type_inference(ast_node: ConditionalASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    condition_inferred_type = determine_expression_ast_node_type(ast_node.condition, var_name_to_type_info)
    assert_type_consistency(ast_node, condition_inferred_type, TensorTypeASTNode(base_type_name='Boolean', shape=[]))
    var_name_to_type_info, changed = perform_type_inference(ast_node.then_body, var_name_to_type_info, latest_function_name)
    if ast_node.else_body is not None:
        var_name_to_type_info, else_changed = perform_type_inference(ast_node.else_body, var_name_to_type_info, latest_function_name)
        changed |= else_changed
    return var_name_to_type_info, changed

@register_type_inference_method(WhileLoopASTNode)
def while_loop_type_inference(ast_node: WhileLoopASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    condition_inferred_type = determine_expression_ast_node_type(ast_node.condition, var_name_to_type_info)
    assert_type_consistency(ast_node, condition_inferred_type, TensorTypeASTNode(base_type_name='Boolean', shape=[]))
    var_name_to_type_info, changed = perform_type_inference(ast_node.body, var_name_to_type_info, latest_function_name)
    return var_name_to_type_info, changed

@register_type_inference_method(ForLoopASTNode)
def for_loop_type_inference(ast_node: ForLoopASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    changed = False
    inferred_types = [
        determine_expression_ast_node_type(ast_node.minimum, var_name_to_type_info),
        determine_expression_ast_node_type(ast_node.supremum, var_name_to_type_info),
        determine_expression_ast_node_type(ast_node.delta, var_name_to_type_info)
    ]
    iterator_variable_known = ast_node.iterator_variable_name in var_name_to_type_info
    if iterator_variable_known:
        inferred_types.append(var_name_to_type_info[ast_node.iterator_variable_name])
    assert_type_consistency(ast_node, *inferred_types)
    if iterator_variable_known:
        var_name_to_type_info, changed = perform_type_inference(ast_node.body, var_name_to_type_info, latest_function_name)
    else:
        inferred_type = only_one(quadratic_unique(inferred_types))
        var_name_to_type_info[ast_node.iterator_variable_name] = inferred_type
        var_name_to_type_info, changed = perform_type_inference(ast_node.body, var_name_to_type_info, latest_function_name)
        del var_name_to_type_info[ast_node.iterator_variable_name]
    return var_name_to_type_info, changed

@register_type_inference_method(ScopedStatementSequenceASTNode)
def scoped_statement_sequence_type_inference(ast_node: ScopedStatementSequenceASTNode, var_name_to_type_info: Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], latest_function_name: Optional[str]) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    changed = False
    sequece_var_name_to_type_info = var_name_to_type_info
    for statement in ast_node.statements:
        sequece_var_name_to_type_info, _ = perform_type_inference(statement, sequece_var_name_to_type_info, latest_function_name)
        # only update variables known outside of scope
        for var_name in var_name_to_type_info.keys():
            changed = var_name_to_type_info[var_name] != sequece_var_name_to_type_info[var_name]
            if changed:
                var_name_to_type_info[var_name] = sequece_var_name_to_type_info[var_name]
    return var_name_to_type_info, changed

assert set(child_classes(ASTNode)) == set(AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD.keys())

###############
# Entry Point #
###############

def perform_type_inference(
        ast_node: ASTNode,
        var_name_to_type_info: Optional[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]]] = None,
        latest_function_name: Optional[str] = None
) -> Tuple[Dict[str, Union[TensorTypeASTNode, FunctionDefinitionASTNode]], bool]:
    '''
    Modifies the AST node in place to insert inferred types.
    Runs until quiescence.
    Checks type consistency and raises exceptions upon inconsistency.
    '''
    if var_name_to_type_info is None:
        var_name_to_type_info = dict()
    ast_node_type = type(ast_node)
    type_inference_method = AST_NODE_TYPE_TO_TYPE_INFERENCE_METHOD[ast_node_type]
    var_name_to_type_info, changed = type_inference_method(ast_node, var_name_to_type_info, latest_function_name)
    while changed:
        var_name_to_type_info, changed = type_inference_method(ast_node, var_name_to_type_info, latest_function_name)
    return var_name_to_type_info, changed

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")
