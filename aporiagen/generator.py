from collections import defaultdict
from aporia import aporia_ast
from aporia.parser import parser
from aporiagen.counter import count_objects
import random

class Generator:
    def __init__(self, program:str=None, num_stmts:int=None):
        if not program and not num_stmts:
            raise ValueError("Must provide either program or num_instr")
        if program:
            self.program_ast = parser.parse(program)
            self.count = count_objects(self.program_ast)
            self.num_instr = self.count[aporia_ast.Stmt]
        else:
            self.num_instr = num_stmts
        self.type_to_vars = defaultdict(set)
        self.num_vars = 0

    def run(self):
        statements = []
        self.type_to_vars = defaultdict(set)
        self.num_vars = 0
        for i in range(self.num_instr):
            statements.append(self.generate_stmt(aporia_ast.Stmt))
        declarations = []
        for type, variables in self.type_to_vars.items():
            variables = {aporia_ast.Var(v) for v in variables}
            if len(variables) > 0:
                declarations.append(aporia_ast.Declar(type(), variables))
        return aporia_ast.L_cfi(declarations, statements)

    def generate_variable(self):
        self.num_vars += 1
        return f"var_{self.num_vars}"

    def generate_stmt(self, stmt):
        types = [aporia_ast.Int, aporia_ast.Bool, aporia_ast.Float]
        match stmt:
            case aporia_ast.Stmt:
                pred = self.generate_stmt(aporia_ast.Pred)
                inst = self.generate_stmt(aporia_ast.Inst)
                assert inst is not None
                return aporia_ast.Stmt(None, pred, inst)
            case aporia_ast.Inst:
                return self.generate_stmt(random.choice(aporia_ast.Inst.__subclasses__()))
            case aporia_ast.Pred:
                exp_type = random.choice((aporia_ast.Bools, aporia_ast.Var))
                if exp_type == aporia_ast.Bools or len(self.type_to_vars[aporia_ast.Bool]) == 0:
                    return aporia_ast.Bools(True)
                exp = self.generate_expr(exp_type, aporia_ast.Bool)
                return aporia_ast.Pred(exp)
            case aporia_ast.PrintInst:
                type = random.choice(types)
                exp = self.generate_expr(aporia_ast.Exp, type)
                return aporia_ast.PrintInst("", exp)
            case aporia_ast.AssignInst:
                var_to_type = {var: t for t, vars_set in self.type_to_vars.items() for var in vars_set}
                if random.random() < 0.5 and len(var_to_type) > 0:
                    var = random.choice(list(var_to_type.keys()))
                    type = var_to_type[var]
                    exp = self.generate_expr(aporia_ast.Exp, type)
                else:
                    name = self.generate_variable()
                    type = random.choice(types)
                    var = aporia_ast.Var(name)
                    exp = self.generate_expr(aporia_ast.Exp, type)
                    self.type_to_vars[type].add(name)
                return aporia_ast.AssignInst(aporia_ast.Assign(var, exp))
            case aporia_ast.ExpInst:
                type = random.choice(types)
                exp = self.generate_expr(aporia_ast.Exp, type)
                return aporia_ast.ExpInst(exp)
            case _:
                raise Exception("Unexpected input " + repr(stmt))

    def generate_expr(self, expr, type):
        match expr:
            case aporia_ast.ExpInst:
                exp = self.generate_expr(aporia_ast.Exp, type)
                return aporia_ast.ExpInst(exp)
            case aporia_ast.Exp:
                exp = [aporia_ast.Var, aporia_ast.BinOp, aporia_ast.UnaryOp,
                       aporia_ast.Bools if type == aporia_ast.Bool else aporia_ast.Constant]
                return self.generate_expr(random.choice(exp), type)
            case aporia_ast.Var:
                if len(self.type_to_vars[type]) == 0:
                    return self.generate_expr(aporia_ast.Bools if type == aporia_ast.Bool else aporia_ast.Constant, type)
                name = random.choice(list(self.type_to_vars[type]))
                return aporia_ast.Var(name)
            case aporia_ast.Constant:
                assert type != aporia_ast.Bool
                value = random.choice(list(range(1, 10)))
                if type == aporia_ast.Float:
                    value = float(value)
                return aporia_ast.Constant(value)
            case aporia_ast.Bools:
                assert type != aporia_ast.Int or type != aporia_ast.Float
                value = random.choice((True, False))
                return aporia_ast.Bools(value)
            case aporia_ast.UnaryOp:
                exp = self.generate_expr(aporia_ast.Exp, type)
                if type == aporia_ast.Bool:
                    op = random.choice(list(aporia_ast.UnaryBoolOperator.__subclasses__()))
                else:
                    op = random.choice(list(aporia_ast.UnaryNumOperator.__subclasses__()))
                return aporia_ast.UnaryOp(op(), exp)
            case aporia_ast.BinOp:
                left = self.generate_expr(aporia_ast.Exp, type)
                right = self.generate_expr(aporia_ast.Exp, type)
                if type == aporia_ast.Bool:
                    if random.random() < 0.5:
                        cmp = random.choice(aporia_ast.Comparator.__subclasses__())
                        type = random.choice((aporia_ast.Int, aporia_ast.Float))
                        left = self.generate_expr(aporia_ast.Exp, type)
                        right = self.generate_expr(aporia_ast.Exp, type)
                        return aporia_ast.BinOp(left, cmp(), right)
                    op_types = list(aporia_ast.BinaryBoolOperator.__subclasses__())
                elif type == aporia_ast.Int:
                    op_types = [aporia_ast.Add, aporia_ast.Sub, aporia_ast.Mult, aporia_ast.FloorDiv, aporia_ast.Mod]
                else:
                    op_types = [aporia_ast.Add, aporia_ast.Sub, aporia_ast.Mult, aporia_ast.Div]
                op = random.choice(op_types)
                return aporia_ast.BinOp(left, op(), right)
            case _:
                raise Exception("Unexpected input " + repr(expr))