
'''
'''

# TODO fill in doc string

###########
# Imports #
###########

import typing
import typing_extensions
import weakref
import operator
import pyparsing

from .misc_utilities import *

# TODO make sure imports are used
# TODO make sure these imports are ordered in some way

#######################################
# Base Type Sanity Checking Utilities #
#######################################

BASE_TYPES = ('Boolean', 'Integer', 'Float', 'NothingType')

BaseTypeName = operator.getitem(typing_extensions.Literal, BASE_TYPES)

class BaseTypeTrackerType(type):
    
    instantiated_base_type_tracker_classes: typing.List[weakref.ref] = []
    
    def __new__(meta, class_name: str, bases: typing.Tuple[type, ...], attributes: dict) -> type:
        
        updated_attributes = dict(attributes)
        assert 'tracked_type' in updated_attributes
        updated_attributes['base_type_to_value'] = {}
        result_class = type.__new__(meta, class_name, bases, updated_attributes)
        
        result_class_weakref = weakref.ref(result_class)
        meta.instantiated_base_type_tracker_classes.append(result_class_weakref)
        
        return result_class
    
    def __getitem__(cls, base_type_name: BaseTypeName) -> typing.Callable[[typing.Any], typing.Any]:
        def note_value(value: typing.Any) -> typing.Any:
            assert base_type_name not in cls.base_type_to_value
            assert isinstance(value, cls.tracked_type), f'{value} is not an instance of the tracked type {cls.tracked_type}'
            cls.base_type_to_value[base_type_name] = value
            return value
        return note_value
    
    def __getattr__(cls, base_type_name: BaseTypeName) -> typing.Callable[[typing.Any], typing.Any]:
        return cls[base_type_name]
    
    @classmethod
    def vaildate_base_types(meta) -> None:
        for instantiated_base_type_tracker_class_weakref in meta.instantiated_base_type_tracker_classes:
            instantiated_base_type_tracker_class = instantiated_base_type_tracker_class_weakref()
            instantiated_base_type_tracker_class_is_alive = instantiated_base_type_tracker_class is not None
            assert instantiated_base_type_tracker_class_is_alive
            assert len(BASE_TYPES) == len(instantiated_base_type_tracker_class.base_type_to_value.keys())
            assert all(key in BASE_TYPES for key in instantiated_base_type_tracker_class.base_type_to_value.keys())
            assert all(base_type in instantiated_base_type_tracker_class.base_type_to_value.keys() for base_type in BASE_TYPES)
        return

def sanity_check_base_types() -> None:
    BaseTypeTrackerType.vaildate_base_types()
    return 

class TensorTypeParserElementBaseTypeTracker(metaclass=BaseTypeTrackerType):
    tracked_type: type = pyparsing.ParserElement

class AtomicLiteralParserElementBaseTypeTracker(metaclass=BaseTypeTrackerType):
    tracked_type: type = pyparsing.ParserElement

class LiteralASTNodeClassBaseTypeTracker(metaclass=BaseTypeTrackerType):
    tracked_type: type = type
