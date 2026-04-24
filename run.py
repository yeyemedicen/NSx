from nsx import solver
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)


def run(inputfile):
    ''' Run Navier-Stokes simulation with the configuration specified in the
    given input file.

    Args:
        inputfile (str):   YAML input file with problem and solver
            configuration
    '''
    sol = solver.init(inputfile)
    sol.solve()


def get_parser():
    ''' Get arguments parser '''
    parser = argparse.ArgumentParser(
        description='''\
Run Navier-Stokes simulation with the configuration specified in the given
input file.''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputfile', type=str, help='path to YAML input file')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    run(args.inputfile)
