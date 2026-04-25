
from common.inout import read_parameters

def init(inputfile):
    ''' Select solver according to inputfile and return solver instance.

    Args:
        inputfile (str):   path/to/inputfile.yaml

    Returns:
        NavierStokes Solver object
    '''
    options = read_parameters(inputfile)

    scheme = options['timemarching']['velocity_pressure_coupling']
    
    if scheme == 'monolithic':
        from nsx.monolithic.problem import problem
        from nsx.monolithic.solver import solver
    elif scheme == 'fractionalstep':
        from nsx.fractionalstep.problem import problem
        from nsx.fractionalstep.solver import solver
    else:
        raise Exception('Velocity--pressure coupling scheme \'{}\' not '
                        'recognized. Available options: [monolithic, '
                        'fractionalstep]'.format(scheme))

    problem = problem(inputfile)
    problem.init()
    solver = solver(problem)

    return solver
