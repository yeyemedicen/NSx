from dolfin import *
import dolfin
from ..logger.logger import LoggerBase

__all__ = ['SDParameter', 'Metric']


class SDParameter(LoggerBase):
    def __init__(self, options, mesh, mu, rho, k, logging_filehandler):
        super().__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.addHandler(ch)
        self._logging_filehandler = logging_filehandler
        if self._logging_filehandler:
            self.logger.addHandler(self._logging_filehandler)

        self.options = options
        self.mesh = mesh
        self.mu = mu
        self.rho = rho
        self.k = k

    def stabilization_parameter(self, u_conv, u_conv_assigned=None):
        ''' SD stabilization parameter.

        Args:
            u_conv  convecting velocity (vector function or as_vector(u_lst)
                    for component wise treatment)
            u_conv_assigned     convecting velocity vector function (assigned)
                    when using component splitting

        Returns:
            tau     stabilization parameter (ufl function)
        '''
        parameter = (self.options['fem']['stabilization']
                     ['streamline_diffusion']['parameter'])

        if u_conv_assigned is None:
            u_conv_assigned = u_conv

        if parameter == 'shakib':
            tau = self.tau_shakib(u_conv, u_conv_assigned)
        elif parameter in ('standard', 'default'):
            tau = self.tau_standard(u_conv, u_conv_assigned)
        elif parameter == 'klr':
            tau = self.tau_klr(u_conv, u_conv_assigned)
        elif parameter == 'codina':
            tau = self.tau_codina(u_conv, u_conv_assigned)
        else:
            raise Exception('streamline diffusion parameter unknown: {}'.
                            format(parameter))

        return tau

    def tau_shakib(self, u_conv, u_conv_assigned):
        opt = self.options['fem']['stabilization']['streamline_diffusion']
        # if (self.options['timemarching']['fractionalstep']['scheme'] ==
        #         'IPCS'):
        #     alpha = 2.
        # else:
        #     alpha = 1.
        # FIXME: need reference for this! does alpha depend on theta?
        alpha = Constant(2.)

        rho = self.rho
        mu = self.mu
        k = self.k

        if opt['length_scale'] == 'metric':
            G = Metric(self.mesh).create()
            cinv_default = 30.
            len_scale = 'metric'
        elif opt['length_scale'] == 'max':
            h = MaxCellEdgeLength(self.mesh)
            cinv_default = 12.
            len_scale = 'max'
        else:
            h = CellDiameter(self.mesh)
            cinv_default = 12.
            len_scale = 'average'

        if isinstance(opt['Cinv'], (int, float)):
            Cinv = Constant(opt['Cinv'])
        else:
            Cinv = Constant(cinv_default)

        if not ('parameter_element_constant' in opt and
                opt['parameter_element_constant']):

            if opt['length_scale'] == 'metric':
                tau = (
                    (alpha*k)**2 + dot(u_conv, G*u_conv)
                    + Cinv*(mu/rho)**2*inner(G, G)
                )**(-0.5)
            else:
                tau = (
                    (alpha*k)**2 + (2/h)**2*dot(u_conv, u_conv)
                    + (Cinv*mu/rho/h**2)**2
                )**(-0.5)

        else:

            if opt['length_scale'] == 'metric':
                tau_cpp_code = _tau_cpp_shakib_metric
            elif opt['length_scale'] == 'average':
                tau_cpp_code = _tau_cpp_shakib_average

            if (dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                                 and has_pybind11())):
                tau = CompiledExpression(compile_cpp_code(tau_cpp_code).tau(),
                                         element=FiniteElement(
                                             'DG', self.mesh.ufl_cell(), 0),
                                         domain=self.mesh
                                         # mpi_comm=self.mesh.mpi_comm()
                                         )
                tau.viscosity = mu.cpp_object()
                tau.density = rho.cpp_object()
                tau.k = k.cpp_object()
                # tau.alpha = alpha.cpp_object()
                tau.alpha = float(alpha)
                tau.cinv = Cinv.cpp_object()
                tau.u = u_conv_assigned.cpp_object()

                if opt['length_scale'] == 'metric':
                    tau.G = G.cpp_object()
            else:
                tau = Expression(tau_cpp_code,
                                 element=FiniteElement(
                                     'DG', self.mesh.ufl_cell(), 0),
                                 domain=self.mesh,
                                 mpi_comm=self.mesh.mpi_comm()
                                 )
                tau.viscosity = mu
                tau.density = rho
                tau.k = k
                tau.alpha = float(alpha)
                tau.cinv = Cinv
                tau.u = u_conv_assigned

                if opt['length_scale'] == 'metric':
                    tau.G = G

        self.logger.info('SD parameter: {p}\n\tlength scale: {ls}\n'
                         '\tC_inv: {c}'.format(p=opt['parameter'],
                                               ls=len_scale, c=float(Cinv)))

        return tau

    def tau_standard(self, u_conv, u_conv_assigned):
        opt = self.options['fem']['stabilization']['streamline_diffusion']
        if opt['length_scale'] == 'metric':
            len_scale = 'metric'
            tau_cpp_code = _tau_cpp_standard_metric
        else:
            len_scale = 'average'
            tau_cpp_code = _tau_cpp_standard_average

        if (dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                             and has_pybind11())):

            tau = CompiledExpression(compile_cpp_code(tau_cpp_code).tau(),
                                     element=FiniteElement(
                                         'DG', self.mesh.ufl_cell(), 0),
                                     domain=self.mesh
                                     )
            tau.viscosity = self.mu.cpp_object()
            tau.density = self.rho.cpp_object()
            tau.u = u_conv_assigned.cpp_object()

            if opt['length_scale'] == 'metric':
                tau.G = Metric(self.mesh).create().cpp_object()

        else:

            tau = Expression(tau_cpp_code, element=FiniteElement(
                             'DG', self.mesh.ufl_cell(), 0),
                             domain=self.mesh,
                             mpi_comm=self.mesh.mpi_comm())
            tau.viscosity = self.mu
            tau.density = self.rho
            tau.u = u_conv_assigned

            if opt['length_scale'] == 'metric':
                tau.G = Metric(self.mesh).create()

        self.logger.info('SD parameter: {p}\n\tlength scale: {ls}\n'
                         .format(p=opt['parameter'], ls=len_scale))

        return tau

    def tau_klr(self, u_conv, u_conv_assigned):
        if (dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                             and has_pybind11())):
            tau = CompiledExpression(compile_cpp_code(_tau_cpp_klr).tau(),
                                     element=FiniteElement(
                                         'DG', self.mesh.ufl_cell(), 0),
                                     domain=self.mesh
                                     )
            tau.viscosity = self.mu.cpp_object()
            tau.density = self.rho.cpp_object()
            tau.dt_inv = self.k.cpp_object()
            tau.pk = Constant(self.Vi.ufl_element().degree()**2).cpp_object()
            tau.u = u_conv_assigned.cpp_object()

        else:

            tau = Expression(_tau_cpp_klr,
                             element=FiniteElement('DG', self.mesh.ufl_cell(),
                                                   0),
                             domain=self.mesh, mpi_comm=self.mesh.mpi_comm())
            tau.pk = Constant(self.Vi.ufl_element().degree()**2)
            tau.viscosity = self.mu
            tau.density = self.rho
            tau.dt_inv = self.k
            tau.u = u_conv_assigned

        self.logger.info('SD parameter: KLR')

        return tau

    def tau_codina(self, u_conv, u_conv_assigned):
        h = CellDiameter(self.mesh)
        rho = self.rho
        mu = self.mu
        k = self.k
        tau = 1./(4*mu/rho/h**2 + sqrt(inner(u_conv, u_conv))/h + 1.5*k)

        self.logger.info('SD parameter: Codina')

        return tau


class Metric:
    def __init__(self, mesh):
        self.mesh = mesh

    def create(self):
        return self.metric_cpp()

    def metric_py(self):
        ''' Return metric G of elements '''
        raise DeprecationWarning('Metric via python Expression subclassing '
                                 'not supported anymore!')
        DIM = self.mesh.topology().dim()

        class Metric(Expression):
            def __init__(self, mesh, **kwargs):
                self.mesh = mesh
                self.dim = mesh.topology().dim()

            def eval_cell(self, values, x, cell):
                x = Cell(self.mesh, cell.index).get_vertex_coordinates()
                if self.dim == 2:
                    x0, x1, x2 = x.reshape(3, 2)
                    # [x, = x0*(1-s-t) + x1*s + x2*t = x0 + [x1-x0 x2-x0][s,
                    #  y]                                                 t]
                    F = np.c_[x1 - x0, x2 - x0]   # d(x, y)/d(s, t)
                elif self.dim == 3:
                    x0, x1, x2, x3 = x.reshape(4, 3)
                    F = np.c_[x1 - x0, x2 - x0, x3 - x0]   # d(x, y)/d(s, t)
                Finv = np.linalg.inv(F)
                values[:] = (Finv.T.dot(Finv)).flatten()

            def value_shape(self):
                return (DIM, DIM)

        t0 = Timer('Z metric tensor')
        t0.start()
        G = Metric(self.mesh, degree=0)
        Th = TensorFunctionSpace(self.mesh, 'DG', 0)
        Gh = interpolate(G, Th)
        t0.stop()
        return Gh

    def metric_cpp(self):
        ''' Metric C++ Expression '''
        dim = self.mesh.topology().dim()

        t0 = Timer('Z metric cpp tensor')
        t0.start()

        if (dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                             and has_pybind11())):
            cppcode = '''
    #include <pybind11/pybind11.h>
    namespace py = pybind11;

    #include <Eigen/Dense>
    #include <dolfin/function/Expression.h>
    #include <dolfin/mesh/Mesh.h>
    #include <dolfin/mesh/Cell.h>

    class Metric : public dolfin::Expression
    {{
        public:
            std::shared_ptr<const dolfin::Mesh> mesh;

            Metric() : dolfin::Expression({D}, {D}) {{ }}

        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c)
                const override
        {{
            const dolfin::Cell cell(*mesh, c.index);
            std::vector<double> coord;
            cell.get_vertex_coordinates(coord);
            Eigen::Map<Eigen::Matrix<double, {D1}, {D}, Eigen::RowMajor> > \
                XC(coord.data());
            Eigen::Matrix<double, {D}, {D}, Eigen::RowMajor> F, Finv;
            for (uint i=0; i < {D}; ++i)
                F.col(i) = XC.row(i+1) - XC.row(0);
            Finv = F.inverse();
            F = Finv.transpose() * Finv;
            values = Eigen::Map<Eigen::VectorXd>(F.data(), {DD});
        }}
    }};

    PYBIND11_MODULE(SIGNATURE, m)
    {{
        py::class_<Metric, std::shared_ptr<Metric>, dolfin::Expression>
        (m, "Metric")
        .def(py::init<>())
        .def_readwrite("mesh", &Metric::mesh);
    }}
    '''
            G = CompiledExpression(
                compile_cpp_code(cppcode.format(D=dim, D1=dim+1,
                                                DD=dim**2)).Metric(),
                element=TensorElement('DG', self.mesh.ufl_cell(), 0)
            )

        else:

            cppcode = '''
    class Metric : public Expression
    {{
        public:
            std::shared_ptr<const Mesh> mesh;

            Metric() : Expression({D}, {D}) {{ }}

        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c)
                const override
        {{
            const Cell cell(*mesh, c.index);
            std::vector<double> coord;
            cell.get_vertex_coordinates(coord);
            Eigen::Map<Eigen::Matrix<double, {D1}, {D}, Eigen::RowMajor> > \
                XC(coord.data());
            Eigen::Matrix<double, {D}, {D}, Eigen::RowMajor> F, Finv;
            for (uint i=0; i < {D}; ++i)
                F.col(i) = XC.row(i+1) - XC.row(0);
            Finv = F.inverse();
            F = Finv.transpose() * Finv;
            values = Eigen::Map<Eigen::VectorXd>(F.data(), {DD});
        }}
    }};
    '''
            G = Expression(cppcode=cppcode.format(D=dim, D1=dim+1, DD=dim**2),
                           element=TensorElement('DG', self.mesh.ufl_cell(),
                                                 0),
                           mpi_comm=self.mesh.mpi_comm())

        G.mesh = self.mesh
        Th = TensorFunctionSpace(self.mesh, 'DG', 0)
        Gh = interpolate(G, Th)
        t0.stop()
        return Gh


if (dolfin.__version__ >= '2018' or (hasattr(dolfin, 'has_pybind11')
                                     and has_pybind11())):
    _tau_cpp_shakib_metric = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<dolfin::GenericFunction> u;
    std::shared_ptr<dolfin::GenericFunction> G;
    double alpha;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
//        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        dolfin::Array<double> dt_inv(k->value_size());
        dolfin::Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        // Compute l2 norm of velocity
        double u_inner_metric = 0.;
        double v = 0.;
        dolfin::Array<double> w(u->value_size());
        dolfin::Array<double> g(G->value_size());
        u->eval(w, x, c);
        G->eval(g, x, c);
        for (uint i = 0; i < w.size(); ++i)
        {
            v = 0.;
            for (uint j = 0; j < w.size(); ++j)
                v += g[i*w.size()+j]*w[j];
                // u_inner += w[i]*g[i][j]*w[j];
            u_inner_metric += w[i]*v;
        }

        double gg = 0.;
        for (uint i = 0; i < g.size(); ++i)
            gg += g[i]*g[i];

        values[0] = 1./sqrt(pow(alpha*dt_inv[0], 2) + \
                            u_inner_metric + \
                            Cinv[0]*pow(mu[0]/rho[0], 2)*gg);
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("G", &tau::G)
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity)
    .def_readwrite("k", &tau::k)
    .def_readwrite("alpha", &tau::alpha)
    .def_readwrite("cinv", &tau::cinv);
}
'''

    _tau_cpp_shakib_average = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>


class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<dolfin::GenericFunction> u;
    double alpha;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        const Cell cell(*mesh, c.index);
        double h = cell.h();
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        dolfin::Array<double> dt_inv(k->value_size());
        dolfin::Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        double u_inner = 0.;
        dolfin::Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_inner += w[i]*w[i];

        values[0] = 1./sqrt(pow(alpha*dt_inv[0], 2) + \
                            pow(2/h, 2)*u_inner + \
                            pow(Cinv[0]*mu[0]/rho[0]/(h*h), 2));
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity)
    .def_readwrite("k", &tau::k)
    .def_readwrite("alpha", &tau::alpha)
    .def_readwrite("cinv", &tau::cinv);
}
'''

    _tau_cpp_standard_metric = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density;
    std::shared_ptr<dolfin::GenericFunction> u;
    std::shared_ptr<dolfin::GenericFunction> G;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
        // Evaluate viscosity at given coordinates
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);

        double u_norm2 = 0.;
        double u_inner_metric = 0.;
        double v = 0.;
        dolfin::Array<double> w(u->value_size());
        dolfin::Array<double> g(G->value_size());
        u->eval(w, x, c);
        G->eval(g, x, c);
        for (uint i = 0; i < w.size(); ++i)
        {
            v = 0.;
            u_norm2 += w[i]*w[i];
            for (uint j = 0; j < w.size(); ++j)
                v += g[i*w.size()+j]*w[j];
                // u_inner_metric += w[i]*g[i][j]*w[j];
            u_inner_metric += w[i]*v;
        }
        u_inner_metric = sqrt(u_inner_metric);

        // Compute Peclet number and evaluate stabilization parameter
        double Pe = u_norm2/u_inner_metric*rho[0]/mu[0];
        // "critical" formula
        // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
        // "doubly asymptotic" formula
        double xi = (Pe < 3.0) ? Pe/3. : 1.0;
        // avoid division by zero if norm(u) = 0
        values[0] = (u_inner_metric > 0) ? 1./u_inner_metric*xi : 0;
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("G", &tau::G)
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity);
}
'''

    _tau_cpp_standard_average = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/function/FunctionSpace.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density;
    std::shared_ptr<dolfin::GenericFunction> u;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
               const ufc::cell& c) const
    {
        // Get dolfin cell and its diameter
        const std::shared_ptr<const dolfin::Mesh> mesh = u->function_space()->mesh();
        const dolfin::Cell cell(*mesh, c.index);
        double h = cell.h();
        // Evaluate viscosity at given coordinates
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);

        double u_norm = 0.0;
        dolfin::Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_norm += w[i]*w[i];
        u_norm = sqrt(u_norm);

        // Compute Peclet number and evaluate stabilization parameter
        double Pe = 0.5*u_norm*h*rho[0]/mu[0];
        // "critical" formula
        // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
        // "doubly asymptotic" formula
        double xi = (Pe < 3.0) ? Pe/3. : 1.0;
        values[0] = (u_norm > 0) ? 0.5*h/u_norm*xi : 0;
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("u", &tau::u)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity);
}
'''

    _tau_cpp_klr = '''
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Expression.h>
#include <dolfin/common/Array.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/function/FunctionSpace.h>

class tau : public dolfin::Expression
{
public:
    std::shared_ptr<dolfin::GenericFunction> viscosity, density, dt_inv, pk;
    std::shared_ptr<dolfin::GenericFunction> u;

    tau() : dolfin::Expression() { }

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x,
              const ufc::cell& c) const
    {
        // Get dolfin cell and its diameter
        const std::shared_ptr<const dolfin::Mesh> mesh = u->function_space()->mesh();
        const dolfin::Cell cell(*mesh, c.index);
        double h = cell.h();
        // Evaluate viscosity at given coordinates
        dolfin::Array<double> mu(viscosity->value_size());
        dolfin::Array<double> rho(density->value_size());
        dolfin::Array<double> k(dt_inv->value_size());
        dolfin::Array<double> c0(pk->value_size());
        viscosity->eval(mu, x, c);
        pk->eval(c0, x, c);
        density->eval(rho, x, c);
        dt_inv->eval(k, x, c);
        // Compute l2 norm of velocity
        double u_norm = 0.0;
        dolfin::Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_norm += w[i]*w[i];

        u_norm = sqrt(u_norm);

        double eps = 1.0e-10;

        values[0] = std::min(0.5*h/*u_norm + eps), \
            std::min(2./3./k[0], h*h/(c0[0]*mu[0]/rho[0])));
        // FIXME check factor 2/3
    }
};

PYBIND11_MODULE(SIGNATURE, m)
{
    py::class_<tau, std::shared_ptr<tau>, dolfin::Expression>
    (m, "tau")
    .def(py::init<>())
    .def_readwrite("u", &tau::u)
    .def_readwrite("pk", &tau::pk)
    .def_readwrite("dt_inv", &tau::dt_inv)
    .def_readwrite("density", &tau::density)
    .def_readwrite("viscosity", &tau::viscosity);
}
'''

else:

    _tau_cpp_shakib_metric = '''
class tau : public Expression
{
public:
    std::shared_ptr<GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<GenericFunction> u;
    std::shared_ptr<GenericFunction> G;
    double alpha;

    tau() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& c) const
    {
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        Array<double> mu(viscosity->value_size());
        Array<double> rho(density->value_size());
        Array<double> dt_inv(k->value_size());
        Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        // Compute l2 norm of velocity
        double u_inner_metric = 0.;
        double v = 0.;
        Array<double> w(u->value_size());
        Array<double> g(G->value_size());
        u->eval(w, x, c);
        G->eval(g, x, c);
        for (uint i = 0; i < w.size(); ++i)
        {
            v = 0.;
            for (uint j = 0; j < w.size(); ++j)
                v += g[i*w.size()+j]*w[j];
                // u_inner += w[i]*g[i][j]*w[j];
            u_inner_metric += w[i]*v;
        }

        double gg = 0.;
        for (uint i = 0; i < g.size(); ++i)
            gg += g[i]*g[i];

        values[0] = 1./sqrt(pow(alpha*dt_inv[0], 2) + \
                            u_inner_metric + \
                            Cinv[0]*pow(mu[0]/rho[0], 2)*gg);
    }
};
'''

    _tau_cpp_shakib_average = '''
class tau : public Expression
{
public:
    std::shared_ptr<GenericFunction> viscosity, density, k, cinv;
    std::shared_ptr<GenericFunction> u;
    double alpha;

    tau() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& c) const
    {
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        const Cell cell(*mesh, c.index);
        double h = cell.h();
        Array<double> mu(viscosity->value_size());
        Array<double> rho(density->value_size());
        Array<double> dt_inv(k->value_size());
        Array<double> Cinv(cinv->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);
        k->eval(dt_inv, x, c);
        cinv->eval(Cinv, x, c);

        double u_inner = 0.;
        Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_inner += w[i]*w[i];

        values[0] = 1./sqrt(pow(alpha*dt_inv[0], 2) + \
                            pow(2/h, 2)*u_inner + \
                            pow(Cinv[0]*mu[0]/rho[0]/(h*h), 2));
    }
};
'''

    _tau_cpp_standard_metric = '''
class tau : public Expression
{
public:
    std::shared_ptr<GenericFunction> viscosity, density;
    std::shared_ptr<GenericFunction> u;
    std::shared_ptr<GenericFunction> G;

    tau() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& c) const
    {
        // Evaluate viscosity at given coordinates
        Array<double> mu(viscosity->value_size());
        Array<double> rho(density->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);

        double u_norm2 = 0.;
        double u_inner_metric = 0.;
        double v = 0.;
        Array<double> w(u->value_size());
        Array<double> g(G->value_size());
        u->eval(w, x, c);
        G->eval(g, x, c);
        for (uint i = 0; i < w.size(); ++i)
        {
            v = 0.;
            u_norm2 += w[i]*w[i];
            for (uint j = 0; j < w.size(); ++j)
                v += g[i*w.size()+j]*w[j];
                // u_inner_metric += w[i]*g[i][j]*w[j];
            u_inner_metric += w[i]*v;
        }
        u_inner_metric = sqrt(u_inner_metric);

        // Compute Peclet number and evaluate stabilization parameter
        double Pe = u_norm2/u_inner_metric*rho[0]/mu[0];
        // "critical" formula
        // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
        // "doubly asymptotic" formula
        double xi = (Pe < 3.0) ? Pe/3. : 1.0;
        // avoid division by zero if norm(u) = 0
        values[0] = (u_inner_metric > 0) ? 1./u_inner_metric*xi : 0;
    }
};
'''

    _tau_cpp_standard_average = '''
class tau : public Expression
{
public:
    std::shared_ptr<GenericFunction> viscosity, density;
    std::shared_ptr<GenericFunction> u;

    tau() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x,
               const ufc::cell& c) const
    {
        // Get dolfin cell and its diameter
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        const Cell cell(*mesh, c.index);
        double h = cell.h();
        // Evaluate viscosity at given coordinates
        Array<double> mu(viscosity->value_size());
        Array<double> rho(density->value_size());
        viscosity->eval(mu, x, c);
        density->eval(rho, x, c);

        double u_norm = 0.0;
        Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_norm += w[i]*w[i];
        u_norm = sqrt(u_norm);

        // Compute Peclet number and evaluate stabilization parameter
        double Pe = 0.5*u_norm*h*rho[0]/mu[0];
        // "critical" formula
        // values[0] = (Pe > 1.0) ? 0.5*h*(1.0 - 1.0/Pe)/u_norm : 0.0;
        // "doubly asymptotic" formula
        double xi = (Pe < 3.0) ? Pe/3. : 1.0;
        values[0] = (u_norm > 0) ? 0.5*h/u_norm*xi : 0;
    }
};
'''

    _tau_cpp_klr = '''
class tau : public Expression
{
public:
    std::shared_ptr<GenericFunction> viscosity, density, dt_inv, pk;
    std::shared_ptr<GenericFunction> u;

    tau() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x,
              const ufc::cell& c) const
    {
        // Get dolfin cell and its diameter
        const std::shared_ptr<const Mesh> mesh = u->function_space()->mesh();
        const Cell cell(*mesh, c.index);
        double h = cell.h();
        // Evaluate viscosity at given coordinates
        Array<double> mu(viscosity->value_size());
        Array<double> rho(density->value_size());
        Array<double> k(dt_inv->value_size());
        Array<double> c0(pk->value_size());
        viscosity->eval(mu, x, c);
        pk->eval(c0, x, c);
        density->eval(rho, x, c);
        dt_inv->eval(k, x, c);
        // Compute l2 norm of velocity
        double u_norm = 0.0;
        Array<double> w(u->value_size());
        u->eval(w, x, c);
        for (uint i = 0; i < w.size(); ++i)
            u_norm += w[i]*w[i];

        u_norm = sqrt(u_norm);

        double eps = 1.0e-10;

        values[0] = std::min(0.5*h/*u_norm + eps), \
            std::min(2./3./k[0], h*h/(c0[0]*mu[0]/rho[0])));
        // FIXME check factor 2/3
    }
};
'''
