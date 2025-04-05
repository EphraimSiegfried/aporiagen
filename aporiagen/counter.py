import random
from collections import defaultdict
from aporia.aporia_ast import *
from aporia import aporia_ast
from aporia.parser import parser

class Counter:

    def __init__(self, program_ast=None, source_code="", file_path=""):
        if source_code:
            self.program_ast = parser.parse(source_code)
        elif file_path:
            with open(file_path) as f:
                source_code = f.read()
            self.program_ast = parser.parse(source_code)
        elif program_ast:
            self.program_ast = program_ast

    def variables(self):
        return {type(d.lcfi_type): len(d.var) for d in self.program_ast.declar}

    def objects(self):
        objects = defaultdict(int)

        def count(obj):
            objects[type(obj)] += 1
            if hasattr(obj, '__dict__'):
                for o in obj.__dict__.values():
                    count(o)

        for statement in self.program_ast.stmt:
            count(statement)

        return objects

    def max_depth(self):
        def count(obj):
            if hasattr(obj, '__dict__'):
                d = []
                for o in obj.__dict__.values():
                    d.append(count(o))
                return max(d) + 1 if d else 1
            return 1
        depths = []
        for statement in self.program_ast.stmt:
            depths.append(count(statement))
        return max(depths)

    def mean_and_variance_of_constants(self):
        constants = []
        def collect_constants(obj):
            if type(obj) is aporia_ast.Constant:
                constants.append(obj.value)
            if hasattr(obj, '__dict__'):
                for o in obj.__dict__.values():
                    collect_constants(o)

        for statement in self.program_ast.stmt:
            collect_constants(statement)
        if not constants:
            return None, None
        mean = sum(constants) / len(constants)
        variance = sum((x - mean) ** 2 for x in constants) / len(constants)
        return mean, variance


if __name__ == "__main__":
    code = """
    true: 13
    true: 14
    """
    c = Counter(source_code=code)

    print(c.max_depth())
    mean , variance = c.mean_and_variance_of_constants()
    print(random.gauss(mean, variance))