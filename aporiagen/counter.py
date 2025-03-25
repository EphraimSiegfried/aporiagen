from collections import defaultdict

from aporia.aporia_ast import *


def count_variables(program:L_cfi):
    return {d.lcfi_type: len(d.var) for d in program.declar}


def count_objects(program):
    objects = defaultdict(int)

    def count(obj):
        objects[type(obj)] += 1
        if hasattr(obj, '__dict__'):
            for o in obj.__dict__.values():
                count(o)

    for statement in program.stmt:
        count(statement)

    return objects

# def count_objects(program:L_cfi):
#     objects = defaultdict(int)
#     def count(obj):
#         objects[type(obj)] += 1
#         match obj:
#             case Stmt(_, _, inst):
#                 count(inst)
#             case PrintInst(_, exp) | AssignInst(Assign(_, exp)) | ExpInst(exp) :
#                 count(exp)
#             case UnaryOp(op, exp):
#                 count(op)
#                 count(exp)
#             case BinOp(left, op, right):
#                 count(op)
#                 count(left)
#                 count(right)
#     for statement in program.stmt:
#         count(statement)
#     return objects