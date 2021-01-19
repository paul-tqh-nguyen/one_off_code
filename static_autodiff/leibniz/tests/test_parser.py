
from leibniz import parser

# @debug_on_error
# def main() -> None:
#     input_string = '''
# 0.3213 1.2 -123 1e2 3

def test_parser():
    expected_input_output_pairs = [
        ('123', 123),
        ('0.123', 0.123),
        ('-0.123', -0.123),
        ('1e2', 100.0),
        ('-1e2', -100.0),
    ]
    for input_string, expected_result in expected_input_output_pairs:
        result = parser.parseSourceCode(input_string).asList()[0]
        assert result is expected_result
