import pytest
import random
from aporiagen.generator import Generator
from aporia.interpreter import InterpLcfi

def test_generated_programs():
    num_progs = 1000  # Number of random programs to test
    num_stmts = 100
    generator = Generator(num_instr=num_stmts)

    for i in range(num_progs):
        random.seed(i)
        program = generator.run()
        try:
            InterpLcfi().interp(program)
        except Exception as e:
            pytest.fail(f"Program with seed {i} caused an error:\n{program}\nError: {e}")
