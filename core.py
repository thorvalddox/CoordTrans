import sympy
from itertools import product
from collections import namedtuple
from functools import lru_cache
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from math import isnan


class CoordTrans:
    def __init__(self, f):
        self.f = f

    @classmethod
    def build(cls, u, v, w, tau):
        def func(x, y, z, t):
            return u(x, y, z, t), v(x, y, z, t), w(x, y, z, t), tau(x, y, z, t)

        return cls(func)

    def __call__(self, x, y, z, t):
        if self.f is None:
            raise NotImplementedError
        return self.f(x, y, z, t)

    def test(self, lst):
        for x, y, z, t in product(lst, repeat=4):
            yield (x, y, z, t), self(x, y, z, t)

    def __repr__(self):
        x, y, z, t = sympy.symbols('x y z t')
        try:
            return '\n'.join(map(str, self(x, y, z, t)))
        except NotImplementedError:
            return ('not implemented')
        except Exception:
            print(repr(sympy.simplify(self(x, y, z, t))))
            raise

    def latex(self, swap=False):
        if not swap:
            x, y, z, t = sympy.symbols('x y z t')
            ot = 'u v w \\tau'.split(' ')
        else:
            x, y, z, t = sympy.symbols('u v w \\tau')
            ot = 'x y z t'.split(' ')
        r = '\\\\'.join('{}&=&{}'.format(l, sympy.latex(sympy.simplify(s))) for l, s in zip(ot, self(x, y, z, t)))
        return (r'\left\{{\begin{{array}}{{lcl}}{}\end{{array}}\right.'.format(r))

    def chain(self, other):
        return CoordTrans(lambda x, y, z, t: self(*other(x, y, z, t)))

    def generate_inverse(self):
        x, y, z, t = sympy.symbols('x y z t')
        u, v, w, tau = sympy.symbols('u v w tau')
        uu, vv, ww, ttau = self(x, y, z, t)
        eqs = [u - uu, v - vv, w - ww, tau - ttau]
        print("equations to solve", eqs)
        sol = sympy.solve(eqs, [x, y, z, t])
        if not isinstance(sol, list):
            sol = [sol]
        for s in sol:
            # print(s)
            ret = []
            for var in [x, y, z, t]:
                def f(xx, yy, zz, tt, var2=var):
                    # print(var2,s[var2])
                    return s[var2].subs({u: xx, v: yy, w: zz, tau: tt})

                ret.append(f)

            yield CoordTrans.build(*ret)


minkovski = sympy.diag(1, 1, 1, -1)


class CoordTransPair():
    def __init__(self, left: CoordTrans, right: CoordTrans):
        self.left = left
        self.right = right

    @classmethod
    def auto(cls, left):
        left = CoordTrans(left)
        return cls(left, next(left.generate_inverse()))

    @classmethod
    def auto_noinv(cls, left):
        left = CoordTrans(left)
        return cls(left, CoordTrans(None))

    def __invert__(self):
        return CoordTransPair(self.right, self.left)

    def __matmul__(self, other):
        return CoordTransPair(self.left.chain(other.left), other.right.chain(self.right))

    def __repr__(self):
        return repr(self.left) + '\n<>\n' + repr(self.right)

    def latex(self):
        try:
            return """{}\Leftrightarrow{}""".format(self.left.latex(), self.right.latex(True))
        except NotImplementedError:
            return """{}\Leftrightarrow ???""".format(self.left.latex())

    def val_unit(self):
        return self.left.chain(self.right)

    def test_inverse(self, lst=[-1, 0, 1, 2]):
        for x, y in self.val_unit().test(lst):
            xx = list(map(float, x))
            yy = list(map(float, y))
            if any(isnan(z) for z in xx + yy):
                print(f'{xx} <{x}> is not {yy} <{y}>')
            else:
                assert xx == yy, f'{xx} <{x}> is not {yy} <{y}>'
        for x, y in (~self).val_unit().test(lst):
            xx = list(map(float, x))
            yy = list(map(float, y))
            if any(isnan(z) for z in xx + yy):
                print(f'{xx} <{x}> is not {yy} <{y}> ~~')
            else:
                assert xx == yy, f'{xx} <{x}> is not {yy} <{y}> ~~'
        print('test successfull')

    def coordswap(self, other):
        return (self @ other @ ~self)

    def transform_eq(self, eq, reverse=False):
        if reverse:
            u, v, w, tau = sympy.symbols('x y z t')
            x, y, z, t = sympy.symbols('u v w \\tau')
            T = self.left(u, v, w, tau)
        else:
            u, v, w, tau = sympy.symbols('u v w \\tau')
            x, y, z, t = sympy.symbols('x y z t')
            T = self.right(u, v, w, tau)
        T = {kk: vv for kk, vv in zip([x, y, z, t], T)}
        return eq.subs(T)

    @lru_cache(maxsize=248)
    def jacobian(self, new_coords=False, inv=False):
        m = []
        if not inv:
            x, y, z, t = sympy.symbols('x y z t')

            for s in self.left(x, y, z, t):
                l = [sympy.diff(s, v) for v in [x, y, z, t]]
                # print(l)
                m.append(l)
        else:
            x, y, z, t = sympy.symbols('u v w \tau')

            for s in self.right(x, y, z, t):
                l = [sympy.diff(s, v) for v in [x, y, z, t]]
                # print(l)
                m.append(l)
        M = sympy.Matrix(m)
        if new_coords:
            M = self.transform_eq(M, inv)
        return sympy.simplify(M)

    @lru_cache(maxsize=248)
    def volume(self, new_coords=False):
        return sympy.simplify(self.jacobian(new_coords).det())

    @lru_cache(maxsize=248)
    def metric(self, new_coords=False):
        try:
            J = self.jacobian(not new_coords, True)
        except NotImplementedError:
            J = self.jacobian(new_coords).inv()

        return sympy.simplify(J.T * minkovski * J)

    def transform_speeds(self, vx, vy, vz):
        return self.jacobian(True)[3:, :]

    def kinetic_energy(self, new_coords=False):
        u, v, w = sympy.symbols('v_u v_v v_w')
        V = sympy.Matrix([[u, v, w, 0]]).T
        return sympy.simplify((V.T * self.metric(new_coords) * V).det())

    def mixed_energy(self, new_coords=False):
        u, v, w = sympy.symbols('v_u v_v v_w')
        V = sympy.Matrix([[u, v, w, 0]]).T
        return sympy.simplify((self.metric(new_coords)[3:, :] * V).det())

    def potential_energy(self, new_coords=False):
        return sympy.simplify(self.metric(new_coords)[3:, 3:].det() + 1)

    def display_physics(self, add_mass=True):
        if add_mass:
            m = sympy.symbols('m')
        else:
            m = 1
        yield '$${}$$'.format(self.latex())
        for new_coords in (True, False):
            try:
                yield 'jacobian:'
                yield '$${}$$'.format(sympy.latex(self.jacobian(new_coords)))
                yield 'volume:'
                yield '$${}$$'.format(sympy.latex(self.volume(new_coords)))
                yield 'metric:'
                yield '$${}$$'.format(sympy.latex(self.metric(new_coords)))
                yield 'kinetic energy:'
                yield '$${}$$'.format(sympy.latex(m * self.kinetic_energy(new_coords)))
                yield 'mixed energy:'
                yield '$${}$$'.format(sympy.latex(m * self.mixed_energy(new_coords)))
                yield 'potential energy:'
                yield '$${}$$'.format(sympy.latex(m * self.potential_energy(new_coords)))
                if new_coords:
                    yield 'relativistic force 1 (space only)'
                    yield '$${}$$'.format(
                        sympy.latex(energy_to_force(m * self.kinetic_energy(new_coords), sympy.symbols('u v w'))))
                    yield 'relativistic force 2 (space and time)'
                    yield '$${}$$'.format(
                        sympy.latex(energy_to_force(m * self.mixed_energy(new_coords), sympy.symbols('u v w'))))
                    yield 'relativistic force 3 (time only)'
                    yield '$${}$$'.format(
                        sympy.latex(energy_to_force(m * self.potential_energy(new_coords), sympy.symbols('u v w'))))
            except NotImplementedError:
                yield 'cannot invert system'
            else:
                break


def energy_to_force(eq, pos_var):
    return sympy.Matrix([[sympy.diff(eq, xx) for xx in pos_var]])


def genCoordTrans(leftgen):
    def dec(*args, **kwargs):
        def left(x, y, z, t):
            kwargs.update(dict(x=x, y=y, z=z, t=t))
            return leftgen(*args, **kwargs)

        return CoordTrans(left)

    return dec


def genCoordTransPair(leftgen):
    def dec(*args, **kwargs):
        def left(x, y, z, t):
            kwargs.update(dict(x=x, y=y, z=z, t=t))
            return leftgen(*args, **kwargs)

        return CoordTransPair.auto(left)

    return dec


def genCoordTransPairSafe(leftgen):
    def dec(*args, **kwargs):
        def left(x, y, z, t):
            kwargs.update(dict(x=x, y=y, z=z, t=t))
            return leftgen(*args, **kwargs)

        return CoordTransPair.auto_noinv(left)

    return dec


def genCoordTransPairPar(leftgen):
    def dec(*args, **kwargs):
        def left(x, y, z, t):
            kwargs.update(dict(x=x, y=y, z=z, t=t))
            return leftgen(*args, inv=False, **kwargs)

        def right(x, y, z, t):
            kwargs.update(dict(x=x, y=y, z=z, t=t))
            return leftgen(*args, inv=True, **kwargs)

        return CoordTransPair(CoordTrans(left), CoordTrans(right))

    return dec


def plot_transform(ct, size, esize, hits_course=9, hits_fine=65, custom=()):
    func = np.vectorize(ct.left)
    course = np.linspace(-size, size, hits_course)
    fine = np.linspace(-size, size, hits_fine)
    zer = np.zeros(fine.shape)

    def draw(X, Y, color):
        xx, yy, _, _ = func(X, Y, zer, zer)
        arr = np.array([xx, yy],dtype=np.complex)
        #print(arr)
        arr = arr[:,~np.isnan(arr).any(axis=0)]
        arr = np.real(arr).astype(np.float)
        plt.plot(arr[0], arr[1], color)

    for c in course:
        color = 'b-' if abs(c) == max(course) else 'k--'
        draw(zer + c, fine,color)
        draw( fine, zer+c, color)

    for c in custom:
        draw( c[0], c[1], 'r-')
    plt.xlim(-esize, esize)
    plt.ylim(-esize, esize)


@genCoordTransPair
def swapleft(x, y, z, t):
    return (y, z, x, t)


@genCoordTransPair
def identity(x, y, z, t):
    return (x, y, z, t)


def get_pgf():
    fakefile = StringIO()
    with fakefile as file:
        plt.savefig(file, format='pgf')
        return fakefile.getvalue().replace('\u2212', '-')


def build_report(ct, c=3, notebook=False):
    t = np.linspace(0, 2 * np.pi, 65)
    X = np.sin(t)
    Y = np.cos(t)
    plt.figure(figsize=(4, 4))
    plot_transform(identity(), 2, 3, custom=[(X, Y)])
    if notebook:
        plt.show()
    else:
        yield get_pgf()
    plt.figure(figsize=(4, 4))
    plot_transform(ct, 2, c, custom=[(X, Y)])
    if notebook:
        plt.show()
    else:
        yield get_pgf()
    plt.figure(figsize=(4, 4))
    plot_transform(swapleft().coordswap(ct), 2, c, custom=[(X, Y)])
    if notebook:
        plt.show()
    else:
        yield get_pgf()
    plt.figure(figsize=(4, 4))
    plot_transform((~swapleft()).coordswap(ct), 2, c, custom=[(X, Y)])
    if notebook:
        plt.show()
    else:
        yield get_pgf()
    yield from ct.display_physics()


def write_report(ct, filename, c=3):
    with open(filename, 'w') as file:
        file.write('\\documentclass{article}\n')
        file.write('\\usepackage{fullpage}\n')
        file.write('\\usepackage{amsmath}\n')
        file.write('\\usepackage{amssymb}\n')
        file.write('\\usepackage{pgf}\n')
        file.write('\\begin{document}\n')
        for line in build_report(ct, c):
            file.write(line)
            file.write('\n\n')
        file.write('\\end{document}')


CoordTransPair.write_report = write_report
ct = SimpleNamespace()
ct.ct = genCoordTrans
ct.auto = genCoordTransPair
ct.param = genCoordTransPairPar
ct.safe = genCoordTransPairSafe
