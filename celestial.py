from core import ct, plt, SimpleNamespace, sympy, build_report, Report_builder


class CelestialBody:
    def __init__(self, parent, dist, rotspeed):
        self.parent = parent
        self.dist = dist
        self.rotspeed = rotspeed

    def get_pos(self, t):
        if self.parent is not None:
            bx, by, bz = self.parent.get_pos(t)
        else:
            bx, by, bz = 0, 0, 0
        omega_t = self.rotspeed * t * 2 * sympy.pi
        return bx + sympy.cos(omega_t) * self.dist, by + sympy.sin(omega_t) * self.dist, bz

    def get_pos_transformed(self, t, trans):
        x, y, z = self.get_pos(t)
        return trans.left(x, y, z, t)

    def display_orbit(self, trans, gran, mx, mbox, *args, **kwargs):
        X = []
        Y = []
        Z = []
        for t in range(gran):
            # print(t/gran*mx)
            x, y, z, t = map(float, self.get_pos_transformed(t / gran * mx, trans))
            if x > mbox or -x > mbox or y > float('Nan') or -y > float('Nan') or z > float('Nan') or -z > float('Nan'):
                x = -float('Nan')
                y = -float('Nan')
                z = -float('Nan')

            # print(x,y,z)
            X.append(x)
            Y.append(y)
            Z.append(z)
        # print(X,Y,Z)
        plt.plot(X, Y, *args, zs=Z, **kwargs)

    def get_transform(self):
        return celestial_transform(self)


@ct.auto
def celestial_transform(cb, dt=0, x=0, y=0, z=0, t=0):
    dx, dy, dz = cb.get_pos(t - dt)
    return (x - dx, y - dy, z - dz, t - dt)


@ct.param
def tilt_axis(angle, dt=0, x=0, y=0, z=0, t=0, inv=False):
    if inv:
        angle = -angle
    return x, y * sympy.cos(angle * 2 * sympy.pi) - z * sympy.sin(angle * 2 * sympy.pi), y * sympy.sin(
        angle * 2 * sympy.pi) + z * sympy.cos(
        angle * 2 * sympy.pi), t


@ct.param
def rotating_frame(speed, x, y, z, t, inv):
    if inv:
        speed = -speed
    return x * sympy.cos(speed * t * 2 * sympy.pi) - y * sympy.sin(speed * t * 2 * sympy.pi), x * sympy.sin(
        speed * t * 2 * sympy.pi) + y * sympy.cos(
        speed * t * 2 * sympy.pi), z, t


def create_true_model():
    ns = SimpleNamespace()
    ns.sun = CelestialBody(None, 0, 0)
    ns.earth = CelestialBody(ns.sun, sympy.symbols('R_{AU}'), sympy.symbols('\\nu_{y}'))
    ns.moon = CelestialBody(ns.earth, sympy.symbols('R_{EM}'), sympy.symbols('\\nu_{m}'))
    ns.year = sympy.symbols('t_{y}')
    ns.month = sympy.symbols('t_{m}')
    return ns


def create_mock_model():
    ns = SimpleNamespace()
    ns.sun = CelestialBody(None, 0, 0)
    ns.earth = CelestialBody(ns.sun, 5, 1 / 5)
    ns.moon = CelestialBody(ns.earth, 2, 1 / 3)
    ns.year = 5
    ns.month = 3
    return ns


def display_model(R, model, transformation, geocentric=True):
    if geocentric:
        transformation = transformation @ rotating_frame(1) @ tilt_axis(1 / 8) @ model.earth.get_transform()
    for x, y in [(model.sun, 'sun'), (model.earth, 'earth'), (model.moon, 'moon')]:
        R.report_text('orbit of ' + y)
        R.report_math(sympy.latex(x.get_pos_transformed(sympy.symbols('t'), transformation)))


def plot_model_geocentric(model, transformation):
    transformation = transformation @ rotating_frame(1) @ tilt_axis(1 / 8) @ model.earth.get_transform()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    model.sun.display_orbit(transformation, model.year * 24 * 2, model.year, 20, '-r')
    model.earth.display_orbit(transformation, 1, 1, 20, 'ob')
    model.moon.display_orbit(transformation, model.year * 24 * 2, model.year, 20, '-g')


def plot_model_heliocentric(model, transformation):
    transformation = transformation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    model.sun.display_orbit(transformation, 1, 1, 20, 'or')
    model.earth.display_orbit(transformation, model.year * 24 * 2, model.year, 20, '-b')
    model.moon.display_orbit(transformation, model.year * 24 * 2, model.year, 20, '-g')


def show_transformation(R, T):
    R.report_text('Free Transformation')
    build_report(R, T)
    R.report_text('Rotating Transformation')
    build_report(R, T @ rotating_frame(1))
    plot_model_geocentric(create_mock_model(), T)
    R.report_graph()
    display_model(R, create_true_model(), T, True)


def write_transformation(T, notebook=True):
    R = Report_builder(notebook)
    show_transformation(R, T)
