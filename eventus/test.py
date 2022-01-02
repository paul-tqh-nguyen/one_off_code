
import tempfile
import os

###########
# Globals #
###########

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BOOK_ANALYZER_SCRIPT_LOCATION = os.path.join(SCRIPT_DIR, 'book_analyzer.py')

#############
# Utilities #
#############

from typing import Tuple
def shell(input_command: str) -> Tuple[str, str, int]:
    '''Handles multi-line input_command'''
    import subprocess
    command = input_command.encode('utf-8')
    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_string, stderr_string = process.communicate(command)
    stdout_string = stdout_string.decode('utf-8')
    stderr_string = stderr_string.decode('utf-8')
    return stdout_string, stderr_string, process.returncode

from typing import Callable
def debug_on_error(func: Callable) -> Callable:
    import pdb
    import traceback
    import sys
    def decorating_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return decorating_function

#########
# Tests #
#########

@debug_on_error
def run_test_simple():
    file_contents = '''
28800538 A b S 44.26 100
28800562 A c B 44.10 100
28800744 R b 100
28800758 A d B 44.18 157
28800773 A e S 44.38 100
28800796 R d 157
28800812 A f B 44.18 157
28800974 A g S 44.27 100
28800975 R e 100
28812071 R f 100
28813129 A h B 43.68 50
28813300 R f 57
28813830 A i S 44.18 100
28814087 A j S 44.18 1000
28814834 R c 100
28814864 A k B 44.09 100
28815774 R k 100
28815804 A l B 44.07 175
28815937 R j 1000
28816245 A m S 44.22 100
'''.strip()
    expected_ans = '''
28800758 S 8832.56
28800796 S NA
28800812 S 8832.56
28800974 B 8865.00
28800975 B NA
28812071 S NA
28813129 S 8806.50
28813300 S NA
28813830 B 8845.00
28814087 B 8836.00
28815804 S 8804.25
28815937 B 8845.00
28816245 B 8840.00
'''
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_file = os.path.join(tmp_dir, 'asdf.txt')
        with open(temp_file, 'w') as f:
            f.write(file_contents)
        cmd = f'cat {temp_file} | python3 {BOOK_ANALYZER_SCRIPT_LOCATION} 200'
        stdout_string, stderr_string, returncode = shell(cmd)
        assert returncode == 0
        assert stderr_string == ''
        assert stdout_string.strip() == expected_ans.strip()
    print('run_test_simple success')
    return

@debug_on_error
def run_test_data(target_size, num_lines):
    with open('./data/book_analyzer.in', 'r') as f:
        file_contents = ''.join(f.readlines()[:num_lines])
        file_contents = file_contents.strip()
    with open(f'./data/book_analyzer.out.{target_size}', 'r') as f:
        expected_ans = ''.join(f.readlines()[:num_lines]).strip()
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_file = os.path.join(tmp_dir, 'asdf.txt')
        with open(temp_file, 'w') as f:
            f.write(file_contents)
        cmd = f'cat {temp_file} | python3 {BOOK_ANALYZER_SCRIPT_LOCATION} {target_size}'
        stdout_string, stderr_string, returncode = shell(cmd)
        assert returncode == 0
        assert stderr_string == ''
        assert expected_ans.startswith(stdout_string.strip())
    print(f'run_test_data {target_size} success')
    return

##########
# Driver #
##########

if __name__ == '__main__':
    # run_test_simple()
    # run_test_data(1, 5500)
    run_test_data(200, 100)
    # run_test_data(10_000, 5500)
