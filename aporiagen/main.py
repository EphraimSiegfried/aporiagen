import argparse
from aporiagen.generator import Generator
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-p", "--path", help="Path to the spp file")
    group.add_argument("-i", "--num-instr", help="Number of instructions to generate", type=int)

    parser.add_argument("-n", "--num-progs", help="Number of programs to generate", type=int, default=1)
    parser.add_argument("-r", "--seed", help="Seed for random generator", type=int)

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    generated_programs = []

    if args.path:
        path = Path(args.path)
        if not path.exists():
            raise FileNotFoundError(f"File {args.path} does not exist")
        with open(args.path) as f:
            source = f.read()
        for _ in range(args.num_progs):
            output = Generator(program=source).run()
            generated_programs.append(str(output))

    if args.num_instr:
        for _ in range(args.num_progs):
            output = Generator(num_instr=args.num_instr).run()
            generated_programs.append(str(output))

    print("\n\n".join(generated_programs))


if __name__ == "__main__":
    main()