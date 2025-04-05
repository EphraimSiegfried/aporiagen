import random
from collections import defaultdict

from aporia.parser import parser
from aporiagen.counter import count_objects
import aporia.aporia_ast as ast


class Generator:
    def __init__(self, program: str = None, num_instr: int = None):
        if not program and not num_instr:
            raise ValueError("Must provide either program or num_instr")
        if program:
            self.program_ast = parser.parse(program)
            self.budget = count_objects(self.program_ast)
            self.num_instr = self.budget[ast.Stmt]
        else:
            self.num_instr = num_instr
            self.budget = defaultdict(lambda: 1000)
        self.type_to_vars = defaultdict(set)
        self.num_vars = 0

    def run(self):
        statements = []
        self.type_to_vars = defaultdict(set)
        self.num_vars = 0

        # Generate Instructions
        for i in range(self.num_instr):
            statements.append(self.generate_stmt(ast.Stmt))

        # Generate Declarations
        declarations = []
        for typ, variables in self.type_to_vars.items():
            variables = {ast.Var(v) for v in variables}
            if len(variables) > 0:
                declarations.append(ast.Declar(typ(), variables))

        generated_prog = ast.L_cfi(declarations, statements)
        return generated_prog

    def generate_variable(self):
        self.num_vars += 1
        return f"var_{self.num_vars}"

    def generate_random_type_expression(self):
        types = [t for t in ast.Type.__subclasses__()]
        random.shuffle(types)
        for t in types:
            exp = self.generate_expr(ast.Exp, t)
            if exp is not None:
                return exp, t
        return None, None

    def generate_stmt(self, obj_type):
        if obj_type not in [ast.Inst] and self.budget[obj_type] == 0:
            return None
        self.budget[obj_type] -= 1
        match obj_type:
            case ast.Stmt:
                pred = self.generate_stmt(ast.Pred)
                if pred is None:
                    self.budget[obj_type] += 1
                    return None
                inst = self.generate_stmt(ast.Inst)
                if inst is None:
                    self.budget[obj_type] += 1
                    return None
                return ast.Stmt(None, pred, inst)
            case ast.Inst:
                inst_types = [c for c in ast.Inst.__subclasses__() if self.budget[c] > 0]
                random.shuffle(inst_types)
                for inst_type in inst_types:
                    inst = self.generate_stmt(inst_type)
                    if inst is not None:
                        return inst
                self.budget[obj_type] += 1
                return None
            case ast.Pred:
                exp_type = random.choice((ast.Bool, ast.Var))
                if exp_type == ast.Bool or len(self.type_to_vars[ast.Bool]) == 0:
                    return ast.Pred(ast.Bools(True))  # TODO: Reconsider this
                exp = self.generate_expr(exp_type, ast.Bool)
                if exp is None:
                    self.budget[obj_type] += 1
                    return None
                return ast.Pred(exp)
            case ast.PrintInst:
                exp, _ = self.generate_random_type_expression()
                if exp is None:
                    self.budget[obj_type] += 1
                    return None
                return ast.PrintInst("", exp)
            case ast.AssignInst:
                var_to_type = {
                    var: t
                    for t, vars_set in self.type_to_vars.items()
                    for var in vars_set
                }
                # 1 is for generating new variables, 2 for reusing them
                choices = ["gen", "reuse"] if len(var_to_type) > 0 else ["gen"]
                random.shuffle(choices)
                for choice in choices:
                    if choice == "gen":
                        name = self.generate_variable()
                        var = ast.Var(name)
                        exp, typ = self.generate_random_type_expression()
                        if exp is not None:
                            self.type_to_vars[typ].add(name)
                            return ast.AssignInst(ast.Assign(var, exp))
                    else:
                        var = random.choice(list(var_to_type.keys()))
                        typ = var_to_type[var]
                        exp = self.generate_expr(ast.Exp, typ)
                        if exp is not None:
                            return ast.AssignInst(ast.Assign(var, exp))
                self.budget[obj_type] += 1
                return None
            case ast.ExpInst:
                exp, _ = self.generate_random_type_expression()
                if exp is None:
                    self.budget[obj_type] += 1
                    return None
                return ast.ExpInst(exp)
            case _:
                raise Exception("Unexpected input " + repr(obj_type))


    def generate_expr(self, expr, output_type):
        if expr != ast.Exp and not self.budget[expr] > 0:
            return None
        self.budget[expr] -= 1
        match expr:
            case ast.Exp:
                exps = [ast.Var, ast.BinOp, ast.UnaryOp, ast.Bools if output_type == ast.Bool else ast.Constant]
                exps = [e for e in exps if self.budget[e] > 0]
                random.shuffle(exps)
                for exp_type in exps:
                    e = self.generate_expr(exp_type, output_type)
                    if e is not None:
                        return e
                self.budget[expr] += 1
                return None
            case ast.Var:
                if len(self.type_to_vars[output_type]) == 0:
                    self.budget[expr] += 1
                    return None
                name = random.choice(list(self.type_to_vars[output_type]))
                return ast.Var(name)
            case ast.Constant:
                if output_type == ast.Bool:
                    self.budget[expr] += 1
                    return None
                # TODO: Make range bigger (maybe calculate mean & variance of input program)
                value = random.choice(list(range(1, 10)))
                if output_type == ast.Float:
                    value = float(value)
                return ast.Constant(value)
            case ast.Bools:
                if output_type == ast.Int or output_type == ast.Float:
                    self.budget[expr] += 1
                    return None
                value = random.choice((True, False))
                return ast.Bools(value)
            case ast.UnaryOp:
                if output_type == ast.Bool:
                    ops = list(ast.UnaryBoolOperator.__subclasses__())
                else:
                    ops = list(ast.UnaryNumOperator.__subclasses__())
                ops = [o for o in ops if self.budget[o] > 0]
                if not ops:
                    self.budget[expr] += 1
                    return None
                op = random.choice(ops)
                self.budget[op] -= 1
                exp = self.generate_expr(ast.Exp, output_type)
                if exp is None:
                    self.budget[expr] += 1
                    self.budget[op] += 1
                    return None
                return ast.UnaryOp(op(), exp)
            case ast.BinOp:
                if output_type == ast.Bool:
                    choices = [ast.Comparator, ast.BinaryBoolOperator]
                    random.shuffle(choices)
                    for choice in choices:
                        bin_op = self.generate_bin_op(choice, output_type)
                        if bin_op is not None:
                            return bin_op
                    self.budget[expr] += 1
                    return None
                else:
                    bin_op = self.generate_bin_op(ast.BinaryNumOperator, output_type)
                    if bin_op is not None:
                        return bin_op
                    self.budget[expr] += 1
                    return None
            case _:
                raise Exception("Unexpected input " + repr(expr))

    def generate_bin_op(self, bin_op_type, output_type):
        def choose_expr(types):
            random.shuffle(types)
            for t in types:
                expr = self.generate_expr(ast.Exp, t)
                if expr is not None:
                    return expr
            return None

        if bin_op_type is ast.Comparator:
            ops = [c for c in ast.Comparator.__subclasses__() if self.budget[c] > 0]
            if not ops:
                return None
            op = random.choice(ops)
            self.budget[op] -= 1
            left = choose_expr([ast.Int, ast.Float])
            right = choose_expr([ast.Int, ast.Float]) if left else None
        elif bin_op_type is ast.BinaryBoolOperator:
            ops = [c for c in ast.BinaryBoolOperator.__subclasses__() if self.budget[c] > 0]
            if not ops:
                return None
            op = random.choice(ops)
            self.budget[op] -= 1
            left = self.generate_expr(ast.Exp, ast.Bool)
            right = self.generate_expr(ast.Exp, ast.Bool) if left else None
        elif bin_op_type is ast.BinaryNumOperator and output_type == ast.Int:
            ops = [o for o in [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod] if self.budget[o] > 0]
            if not ops:
                return None
            op = random.choice(ops)
            self.budget[op] -= 1
            left = self.generate_expr(ast.Exp, ast.Int)
            right = self.generate_expr(ast.Exp, ast.Int) if left else None
        elif bin_op_type is ast.BinaryNumOperator and output_type == ast.Float:
            ops = [o for o in [ast.Add, ast.Sub, ast.Mult, ast.Div] if self.budget[o] > 0]
            if not ops:
                return None
            op = random.choice(ops)
            self.budget[op] -= 1
            # Ensure at least one operand is a float.
            if random.choice([True, False]):
                left = self.generate_expr(ast.Exp, ast.Float)
                right = choose_expr([ast.Int, ast.Float]) if left else None
            else:
                left = choose_expr([ast.Int, ast.Float])
                right = self.generate_expr(ast.Exp, ast.Float) if left else None
        else:
            raise Exception("Unexpected input " + repr(bin_op_type))

        if left is None or right is None:
            self.budget[op] += 1
            return None

        return ast.BinOp(left, op(), right)
