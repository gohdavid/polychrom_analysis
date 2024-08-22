r"""
Array of spherical wells
========================

Confine each subchain in simulation to a sherical well.
Should be used in combination with contrib.initialize_chains.

"""

try:
    import openmm
    import simtk
except Exception:
    import simtk.openmm as openmm


def spherical_well_array(
    sim_object,
    r,
    cell_size,
    particles=None,
    width=1,
    depth=1,
    name="spherical_well_array",
):
    """
    An (array of) spherical potential wells. Uses floor functions to map
    particle positions to the coordinates of the well.

    Parameters
    ----------

    r : float
        Radius of the nucleus
    cell_size : float
        width of cell in lattice of spherical wells
    particles : list of int or np.array
        indices of particles that are attracted
    width : float, optional
        Width of attractive well, nm.
    depth : float, optional
        Depth of attractive potential in kT
        Positive means the walls are repulsive (i.e chain confined within lamina).
        Negative means walls are attractive (i.e. attraction to lamina)
    """

    force = openmm.CustomExternalForce(
        "step(1+d) * step(1-d) * SPHWELLdepth * (1 + cos(3.1415926536*d)) / 2;"
        "d = (sqrt((x1-SPHWELLx)^2 + (y1-SPHWELLy)^2 + (z1-SPHWELLz)^2) - SPHWELLradius) / SPHWELLwidth;"
        "x1 = x - L*floor(x/L);"
        "y1 = y - L*floor(y/L);"
        "z1 = z - L*floor(z/L);"
    )
    force.name = name
    particles = range(sim_object.N) if particles is None else particles
    center = 3 * [cell_size / 2]

    force.addGlobalParameter("SPHWELLradius", r * sim_object.conlen)
    force.addGlobalParameter("SPHWELLwidth", width * sim_object.conlen)
    force.addGlobalParameter("SPHWELLdepth", depth * sim_object.kT)
    force.addGlobalParameter("L", cell_size * sim_object.conlen)
    force.addGlobalParameter("SPHWELLx", center[0] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLy", center[1] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLz", center[2] * sim_object.conlen)

    # adding all the particles on which force acts
    for i in particles:
        # NOTE: the explicit type cast seems to be necessary if we have an np.array...
        force.addParticle(int(i), [])

    return force

def spherical_confinement_array(
    sim_object,
    r,
    cell_size,
    k=5.0,  # How steep the walls are
    invert=False,
    particles=None,
    name="spherical_confinement_array",
):
    """Constrain particles to be within a sphere.
    With no parameters creates sphere with density .3

    Parameters
    ----------
    r : float 
        Radius of confining sphere. 
    k : float, optional
        Steepness of the confining potential, in kT/nm
    density : float, optional, <1
        Density for autodetection of confining radius.
        Density is calculated in particles per nm^3,
        i.e. at density 1 each sphere has a 1x1x1 cube.
    center : [float, float, float]
        The coordinates of the center of the sphere.
    invert : bool
        If True, particles are not confinded, but *excluded* from the sphere.
    particles : list of int
        The list of particles affected by the force.
        If None, apply the force to all particles.
    """

    force = openmm.CustomExternalForce(
        "step(invert_sign*(rad-aa)) * kb * (sqrt((rad-aa)*(rad-aa) + t*t) - t); " 
        "rad = sqrt((x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2 + tt^2);"
        "x1 = x - L*floor(abs(x + rs)/L);" 
        "y1 = y - L*floor(abs(y + rs)/L);"
        "z1 = z - L*floor(abs(z + rs)/L);"
    )
    force.name = name

    particles = range(sim_object.N) if particles is None else particles # N is passed down as N*ncopies
    center = 3 * [0.0] 
    for i in particles:
        force.addParticle(int(i), [])

    if sim_object.verbose:
        print("Spherical confinement with radius = %lf" % r)
    # assigning parameters of the force
    force.addGlobalParameter("kb", k * sim_object.kT / simtk.unit.nanometer)
    force.addGlobalParameter("aa", (r - 1.0 / k) * simtk.unit.nanometer)
    force.addGlobalParameter("t", (1.0 / k) * simtk.unit.nanometer / 10.0)
    force.addGlobalParameter("tt", 0.01 * simtk.unit.nanometer)
    force.addGlobalParameter("invert_sign", (-1) if invert else 1) 
    force.addGlobalParameter("rs", r * 2 * sim_object.conlen)
    force.addGlobalParameter("L", cell_size * sim_object.conlen)
    force.addGlobalParameter("x0", center[0] * simtk.unit.nanometer)
    force.addGlobalParameter("y0", center[1] * simtk.unit.nanometer)
    force.addGlobalParameter("z0", center[2] * simtk.unit.nanometer)

    # TODO: move 'r' elsewhere?..
    sim_object.sphericalConfinementRadius = r

    return force

def flow(params,particle_1,particle_2):
    keys = {'beta', 'R', 'l'}
    assert keys.issubset(params.keys()), f"Some parameters are missing."
    force = openmm.CustomBondForce(
        "beta * (step(R-r) * within + step(r-R) * outside);"
        "within = 2*exp(-R/l)*(l+R)*(-r+l*sinh(r/l))/r;"
        "outside = exp(-(r+R)/l)*(2*exp((r+R)/l)*l*r+exp(2*R/l)*l*(l-R)-(l+2*exp(r/l)*r)*(l+R))/r;"
    )
    force.addGlobalParameter("beta", params["beta"])
    force.addGlobalParameter("R", params["R"])
    force.addGlobalParameter("l", params["l"])
    force.addBond(particle_1, particle_2, [])
    force.name = "flow"
    return force