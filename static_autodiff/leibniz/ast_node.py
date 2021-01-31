
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

from abc import ABC, abstractmethod

from .misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

#####################
# AST Functionality #
#####################

class ASTNode(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def parse_action(cls, s: str, loc: int, tokens: pyparsing.ParseResults) -> 'ASTNode':
        raise NotImplementedError

    def __repr__(self) -> str:
        attributes_string = ', '.join(f'{k}={repr(self.__dict__[k])}' for k in sorted(self.__dict__.keys()))
        return f'{self.__class__.__name__}({attributes_string})'

    def __eq__(self, other: 'ASTNode') -> bool:
        '''Determines syntactic equivalence, not semantic equivalence or canonicality.'''
        raise NotImplementedError

class AtomASTNodeType(type):

    # TODO is this needed or is ExpressionAtomASTNodeType sufficient?

    base_ast_node_class = ASTNode
    
    def __new__(meta, class_name: str, bases: typing.Tuple[type, ...], attributes: dict) -> type:
        method_names = ('__init__', 'parse_action', '__eq__')
        for method_name in method_names:
            assert method_name not in attributes.keys(), f'{method_name} already defined for class {class_name}'

        updated_attributes = dict(attributes)
        # Required Declarations
        value_type: type = updated_attributes.pop('value_type')
        token_checker: typing.Callable[[typing.Any], bool] = updated_attributes.pop('token_checker')
        # Optional Declarations
        value_attribute_name: str = updated_attributes.pop('value_attribute_name', 'value')
        dwimming_func: typing.Callable[[typing.Any], value_type] = updated_attributes.pop('dwimming_func', lambda token: token)
        
        def __init__(self, *args, **kwargs) -> None:
            assert sum(map(len, (args, kwargs))) == 1
            value: value_type = args[0] if len(args) == 1 else kwargs[value_attribute_name]
            assert isinstance(value, value_type)
            setattr(self, value_attribute_name, value)
            return

        @classmethod
        def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> class_name:
            token = only_one(tokens)
            assert token_checker(token)
            value: value_type = dwimming_func(token)
            node_instance: cls = cls(value)
            return node_instance
        
        def __eq__(self, other: ASTNode) -> bool:
            other_value = getattr(other, value_attribute_name, BOGUS_TOKEN)
            same_type = type(self) is type(other)
            
            self_value = getattr(self, value_attribute_name)
            same_value_type = type(self_value) is type(other_value)
            same_value = self_value == other_value
            return same_type and same_value_type and same_value
        
        updated_attributes['__init__'] = __init__
        updated_attributes['parse_action'] = parse_action
        updated_attributes['__eq__'] = __eq__
        
        result_class = type(class_name, bases+(meta.base_ast_node_class,), updated_attributes)
        assert all(hasattr(result_class, method_name) for method_name in method_names)
        return result_class

class StatementASTNode(ASTNode):
    pass

class ExpressionASTNode(StatementASTNode):
    pass

class ExpressionAtomASTNodeType(AtomASTNodeType):
    '''The atomic bases that make up expressions.'''

    base_ast_node_class = ExpressionASTNode

class BinaryOperationExpressionASTNode(ExpressionASTNode):

    def __init__(self, left_arg: ExpressionASTNode, right_arg: ExpressionASTNode) -> None:
        self.left_arg: ExpressionASTNode = left_arg
        self.right_arg: ExpressionASTNode = right_arg
        return

    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'BinaryOperationExpressionASTNode':
        tokens = only_one(group_tokens)
        assert len(tokens) >= 2
        node_instance: cls = reduce(cls, tokens)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.left_arg == other.left_arg and \
            self.right_arg == other.right_arg 

class UnaryOperationExpressionASTNode(ExpressionASTNode):

    def __init__(self, arg: ExpressionASTNode) -> None:
        self.arg: ExpressionASTNode = arg
        return

    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'UnaryOperationExpressionASTNode':
        tokens = only_one(group_tokens)
        token = only_one(tokens)
        node_instance: cls = cls(token)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and self.arg == other.arg

class ArithmeticExpressionASTNode(ExpressionASTNode):
    pass

class BooleanExpressionASTNode(ExpressionASTNode):
    pass

class ComparisonExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> 'ComparisonExpressionASTNode':
        tokens = only_one(group_tokens).asList()
        left_arg, right_arg = tokens
        node_instance: cls = cls(left_arg, right_arg)
        return node_instance

# Literal Node Generation

@LiteralASTNodeClassBaseTypeTracker.Boolean
class BooleanLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = bool
    token_checker = lambda token: token in ('True', 'False')
    dwimming_func = lambda token: True if token == 'True' else False

@LiteralASTNodeClassBaseTypeTracker.Integer
class IntegerLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = int
    token_checker = lambda token: isinstance(token, int)

@LiteralASTNodeClassBaseTypeTracker.Float
class FloatLiteralASTNode(metaclass=ExpressionAtomASTNodeType):
    value_type = float
    token_checker = lambda token: isinstance(token, float)

@LiteralASTNodeClassBaseTypeTracker.NothingType
class NothingTypeLiteralASTNode(ExpressionASTNode):
    
    def __init__(self) -> None:
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'NothingTypeLiteralASTNode':
        assert len(tokens) is 0
        node_instance = cls()
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other)

# Identifier / Variable Node Generation

class VariableASTNode(metaclass=ExpressionAtomASTNodeType):
    value_attribute_name = 'name'
    value_type = str
    token_checker = lambda token: isinstance(token, str)

# Arithmetic Expression Node Generation

class NegativeExpressionASTNode(UnaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    pass

class ExponentExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    pass

class MultiplicationExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    pass

class DivisionExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    pass

class AdditionExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    pass

class SubtractionExpressionASTNode(BinaryOperationExpressionASTNode, ArithmeticExpressionASTNode):
    pass

def parse_multiplication_or_division_expression_pe(_s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> typing.Union[MultiplicationExpressionASTNode, DivisionExpressionASTNode]:
    tokens = only_one(group_tokens).asList()
    node_instance = tokens[0]
    remaining_operation_strings = tokens[1::2]
    remaining_node_instances = tokens[2::2]
    for remaining_operation_string, remaining_node_instance in zip(remaining_operation_strings, remaining_node_instances):
        if remaining_operation_string is '*':
            node_instance = MultiplicationExpressionASTNode(node_instance, remaining_node_instance)
        elif remaining_operation_string is '/':
            node_instance = DivisionExpressionASTNode(node_instance, remaining_node_instance)
        else:
            raise ValueError(f'Unrecognized operation {repr(remaining_operation_string)}')
    return node_instance

def parse_addition_or_subtraction_expression_pe(_s: str, _loc: int, group_tokens: pyparsing.ParseResults) -> typing.Union[AdditionExpressionASTNode, SubtractionExpressionASTNode]:
    tokens = only_one(group_tokens).asList()
    node_instance = tokens[0]
    remaining_operation_strings = tokens[1::2]
    remaining_node_instances = tokens[2::2]
    for remaining_operation_string, remaining_node_instance in zip(remaining_operation_strings, remaining_node_instances):
        if remaining_operation_string is '+':
            node_instance = AdditionExpressionASTNode(node_instance, remaining_node_instance)
        elif remaining_operation_string is '-':
            node_instance = SubtractionExpressionASTNode(node_instance, remaining_node_instance)
        else:
            raise ValueError(f'Unrecognized operation {repr(remaining_operation_string)}')
    return node_instance

# Boolean Expression Node Generation

class NotExpressionASTNode(UnaryOperationExpressionASTNode, BooleanExpressionASTNode):
    pass

class AndExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    pass

class XorExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    pass

class OrExpressionASTNode(BinaryOperationExpressionASTNode, BooleanExpressionASTNode):
    pass

# Comparison Expression Node Generation

class GreaterThanExpressionASTNode(ComparisonExpressionASTNode):
    pass

class GreaterThanOrEqualToExpressionASTNode(ComparisonExpressionASTNode):
    pass

class LessThanExpressionASTNode(ComparisonExpressionASTNode):
    pass

class LessThanOrEqualToExpressionASTNode(ComparisonExpressionASTNode):
    pass

class EqualToExpressionASTNode(ComparisonExpressionASTNode):
    pass

class NotEqualToExpressionASTNode(ComparisonExpressionASTNode):
    pass

comparator_to_ast_node_class = {
    '>': GreaterThanExpressionASTNode,
    '>=': GreaterThanOrEqualToExpressionASTNode,
    '<': LessThanExpressionASTNode,
    '<=': LessThanOrEqualToExpressionASTNode,
    '==': EqualToExpressionASTNode,
    '!=': NotEqualToExpressionASTNode,
}

assert set(comparator_to_ast_node_class.values()) == set(child_classes(ComparisonExpressionASTNode))

def parse_comparison_expression_pe(_s: str, _loc: int, tokens: pyparsing.ParseResults) -> ComparisonExpressionASTNode:
    tokens = tokens.asList()
    assert len(tokens) >= 3
    assert len(tokens) % 2 == 1
    node_instance = comparator_to_ast_node_class[tokens[1]](left_arg=tokens[0], right_arg=tokens[2])
    prev_operand = tokens[2]
    comparators = tokens[3::2]
    operands = tokens[4::2]
    for comparator, operand in zip(comparators, operands):
        cls = comparator_to_ast_node_class[comparator]
        comparison_ast_node = cls(prev_operand, operand)
        node_instance = AndExpressionASTNode(left_arg=node_instance, right_arg=comparison_ast_node)
        prev_operand =  operand
    return node_instance

# Function Call Node Generation

class FunctionCallExpressionASTNode(ExpressionASTNode):
    
    def __init__(self, function_name: str, arg_bindings: typing.List[typing.Tuple[VariableASTNode, ExpressionASTNode]]) -> None:
        self.function_name = function_name
        self.arg_bindings = arg_bindings
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'FunctionCallExpressionASTNode':
        assert len(tokens) is 2
        function_name = tokens[0]
        arg_bindings = eager_map(tuple, map(pyparsing.ParseResults.asList, tokens[1]))
        node_instance = cls(function_name, arg_bindings)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.function_name == other.function_name and \
            all(
                self_arg_binding == other_arg_binding
                for self_arg_binding, other_arg_binding
                in zip(self.arg_bindings, other.arg_bindings)
            )

# Vector Node Generation

class VectorExpressionASTNode(ExpressionASTNode):
    
    def __init__(self, values: typing.List[ExpressionASTNode]) -> None:
        self.values = values
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'VectorExpressionASTNode':
        values = tokens.asList()
        node_instance = cls(values)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            all(
                self_value == other_value
                for self_value, other_value
                in zip(self.values, other.values)
            )

# Tensor Type Node Generation

class TensorTypeASTNode(ASTNode):

    def __init__(self, base_type_name: typing.Optional[BaseTypeName], shape: typing.Optional[typing.List[typing.Union[typing_extensions.Literal['?', '???'], int]]]) -> None:
        '''
        shape == [IntegerLiteralASTNode(value=1), IntegerLiteralASTNode(value=2), IntegerLiteralASTNode(value=3)] means this tensor has 3 dimensions with the given sizes
        shape == [None, IntegerLiteralASTNode(value=2)] means this tensor has 2 dimensions with an unspecified size for the first dimension size and a fixed size for the second dimension 
        shape == [] means this tensor has 0 dimensions, i.e. is a scalar
        shape == None means this tensor could have any arbitrary number of dimensions
        '''
        assert implies(base_type_name is None, shape is None)
        self.base_type_name = base_type_name
        self.shape = shape
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'TensorTypeASTNode':
        assert len(tokens) in (1, 2)
        base_type_name = tokens[0]
        if len(tokens) is 2:
            shape = [None if e=='?' else e for e in tokens[1].asList()]
            if shape == ['???']:
                shape = None
        else:
            shape = []
        node_instance = cls(base_type_name, shape)
        return node_instance
    
    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.base_type_name is other.base_type_name and \
            self.shape == other.shape

# Assignment Node Generation

def parse_variable_type_declaration_pe(_s: str, _loc: int, type_label_tokens: pyparsing.ParseResults) -> TensorTypeASTNode:
    assert len(type_label_tokens) in (0, 1)
    if len(type_label_tokens) is 0:
        node_instance: TensorTypeASTNode = TensorTypeASTNode(None, None)
    elif len(type_label_tokens) is 1:
        node_instance: TensorTypeASTNode = only_one(type_label_tokens)
    else:
        raise ValueError(f'Unexpected type label tokens {type_label_tokens}')
    assert isinstance(node_instance, TensorTypeASTNode)
    return node_instance

class AssignmentASTNode(StatementASTNode):
    
    def __init__(self, variable_type_pairs: typing.List[typing.Tuple[VariableASTNode, TensorTypeASTNode]], value: ExpressionASTNode) -> None:
        self.variable_type_pairs = variable_type_pairs
        self.value = value
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'AssignmentASTNode':
        assert len(tokens) is 2
        variable_type_pairs: typing.List[typing.Tuple[VariableASTNode, TensorTypeASTNode]] = eager_map(tuple, map(pyparsing.ParseResults.asList, tokens[0]))
        value = tokens[1]
        node_instance = cls(variable_type_pairs, value)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.variable_type_pairs == other.variable_type_pairs and \
            self.value == other.value

# Return Statement Node Generation

class ReturnStatementASTNode(StatementASTNode):
    
    def __init__(self, return_values: typing.List[ExpressionASTNode]) -> None:
        self.return_values = return_values
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ReturnStatementASTNode':
        return_values = tokens.asList()
        if len(return_values) is 0:
            return_values = [NothingTypeLiteralASTNode()]
        node_instance = cls(return_values)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.return_values == other.return_values

# Print Statement Node Generation

class PrintStatementASTNode(StatementASTNode):
    
    def __init__(self, values_to_print: typing.List[typing.Union[str, ExpressionASTNode]]) -> None:
        self.values_to_print = values_to_print
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'PrintStatementASTNode':
        values_to_print = tokens.asList()
        node_instance = cls(values_to_print)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.values_to_print == other.values_to_print

# Scoped Statement Sequence Node Generation

class ScopedStatementSequenceASTNode(StatementASTNode):
    
    def __init__(self, statements: typing.List[StatementASTNode]) -> None:
        self.statements = statements
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ScopedStatementSequenceASTNode':
        statements = tokens.asList()
        node_instance = cls(statements)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.statements == other.statements

# Function Definition Node Generation

class FunctionDefinitionASTNode(StatementASTNode):
    
    def __init__(
            self,
            function_name: str,
            function_signature: typing.List[typing.Tuple[VariableASTNode, TensorTypeASTNode]],
            function_return_type: TensorTypeASTNode, 
            function_body: StatementASTNode
    ) -> None:
        self.function_name = function_name
        self.function_signature = function_signature
        self.function_return_type = function_return_type
        self.function_body = function_body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'FunctionDefinitionASTNode':
        function_name, function_signature, function_return_type, function_body = tokens.asList()
        function_signature = eager_map(tuple, function_signature)
        node_instance = cls(function_name, function_signature, function_return_type, function_body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.function_name == other.function_name and \
            self.function_signature == other.function_signature and \
            self.function_return_type == other.function_return_type and \
            self.function_body == other.function_body

# For Loop Node Generation

class ForLoopASTNode(StatementASTNode):
    
    def __init__(
            self,
            iterator_variable_name: str,
            minimum: typing.Union[VariableASTNode, ArithmeticExpressionASTNode],
            supremum: typing.Union[VariableASTNode, ArithmeticExpressionASTNode],
            delta: typing.Union[VariableASTNode, ArithmeticExpressionASTNode],
            body: StatementASTNode
    ) -> None:
        self.iterator_variable_name = iterator_variable_name
        self.minimum = minimum
        self.supremum = supremum
        self.delta = delta
        self.body = body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ForLoopASTNode':
        iterator_variable_name, range_specification, body = tokens.asList()
        assert len(range_specification) in (2, 3)
        if len(range_specification) is 2:
            minimum, supremum = range_specification
            delta = IntegerLiteralASTNode(value=1)
        elif len(range_specification) is 3:
            minimum, supremum, delta = range_specification
        node_instance = cls(iterator_variable_name, minimum, supremum, delta, body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.iterator_variable_name == other.iterator_variable_name and \
            self.minimum == other.minimum and \
            self.supremum == other.supremum and \
            self.delta == other.delta and \
            self.body == other.body

# While Loop Node Generation

class WhileLoopASTNode(StatementASTNode):
    
    def __init__(
            self,
            condition: typing.Union[VariableASTNode, BooleanExpressionASTNode],
            body: StatementASTNode
    ) -> None:
        self.condition = condition
        self.body = body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'WhileLoopASTNode':
        condition, body = tokens.asList()
        node_instance = cls(condition, body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.condition == other.condition and \
            self.body == other.body

# Conditional Node Generation

class ConditionalASTNode(StatementASTNode):
    
    def __init__(
            self,
            condition: typing.Union[VariableASTNode, BooleanExpressionASTNode],
            then_body: StatementASTNode,
            else_body: StatementASTNode,
    ) -> None:
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ConditionalASTNode':
        tokens = tokens.asList()
        assert len(tokens) in (2,3)
        if len(tokens) is 2:
            condition, then_body = tokens
            else_body = None
        elif len(tokens) is 3:
            condition, then_body, else_body = tokens
        node_instance = cls(condition, then_body, else_body)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        return type(self) is type(other) and \
            self.condition == other.condition and \
            self.then_body == other.then_body and \
            self.else_body == other.else_body

# Module Node Generation

class ModuleASTNode(ASTNode):
    
    def __init__(self, statements: typing.List[StatementASTNode]) -> None:
        self.statements: typing.List[StatementASTNode] = statements
        return
    
    @classmethod
    def parse_action(cls, _s: str, _loc: int, tokens: pyparsing.ParseResults) -> 'ModuleASTNode':
        statement_ast_nodes = tokens.asList()
        node_instance = cls(statement_ast_nodes)
        return node_instance

    def __eq__(self, other: ASTNode) -> bool:
        '''Order of statements matters here since this method determines syntactic equivalence.'''
        return type(self) is type(other) and \
            len(self.statements) == len(other.statements) and \
            all(
                self_statement == other_statement
                for self_statement, other_statement
                in zip(self.statements, other.statements)
            )

##########
# Driver #
##########

if __name__ == '__main__':
    print("TODO add something here")
