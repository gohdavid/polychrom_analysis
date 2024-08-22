r"""
Polymer simulations with ActiveBrownianIntegrator
-------------------------------------------------

Python script used to run polychrom simulation with the `ActiveBrownianIntegrator` custom integrator in
contrib.integrators. This integrator is used to simulate a polymer where each mononmer has a different
effective temperature and thus a different diffusion coefficient :math:`D_i = k_B T_i / \xi`. Here, we consider an
the case where there are just two types of monomers, active (A) and inactive (B),
where :math:`D_A > D_B` and the user
chooses the ratio :math:`D_A / D_B`. A/B identities are derived from the Hi-C data in Zhang et
al. (2021). All other parameters are hard-coded and match Goychuk et al. (2023).


"""

import os
import sys
import time

sys.path.append(os.getcwd())

import numpy as np
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter
from simtk import unit

from contrib.integrators import ActiveBrownianIntegrator
from contrib.forces import flow

# 1 is A, 0 is B
ids = np.ones(1000)
ids[799] = 0
N = len(ids)


def run_monomer_diffusion(replicate,
    gpuid, N, ids, activity_ratio, timestep=170, nblocks=1000000, blocksize=100, beta=0
):
    """Run a single simulation on a GPU of a hetero-polymer with A monomers and B monomers. A monomers
    have a larger diffusion coefficient than B monomers, with an activity ratio of D_A / D_B.

    Parameters
    ----------
    gpuid : int
        which GPU to use. If on Mirny Lab machine, should be 0, 1, 2, or 3.
    N : int
        number of monomers in chain
    ids : array-like (N,)
        monomer types. 1 for active (A), 0 for inactive (B)
    activity_ratio : float
        ratio of diffusion coefficient of A monomers to diffusion coefficient of B monomers
    timestep : int
        timestep to feed the Brownian integrator (in femtoseconds)
    nblocks : int
        number of blocks to run the simulation for. For a chain of 1000 monomers, need ~100000 blocks of
        100 timesteps to equilibrate.
    blocksize : int
        number of time steps in a block

    """
    if len(ids) != N:
        raise ValueError(
            "The array of monomer identities must have length equal to the total number of monomers."
        )
    D = np.ones(
        (N, 3)
    )  # Dx, Dy, Dz --> we assume the diffusion coefficient in each spatial dimension is the same
    # by default set the average diffusion coefficient to be 1 kT/friction
    friction_ratio = np.ones(
        (N, 3)
    )
    Bfriction = 25.0
    friction_ratio[ids == 0, :] = Bfriction
    friction_ratio[ids == 1, :] = 1.0
    # monomer density in confinement in units of monomers/volume
    density = 0.224
    r = (3 * N / (4 * 3.141592 * density)) ** (1 / 3)
    print(f"Radius of confinement: {r}")
    timestep = timestep
    # the monomer diffusion coefficient should be in units of kT / friction, where friction = mass*collision_rate
    collision_rate = 2.0
    mass = 100 * unit.amu
    unit_friction = collision_rate * (1.0 / unit.picosecond) * mass
    particlefriction = friction_ratio * unit_friction
    temperature = 300
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature * unit.kelvin
    particleD = unit.Quantity(D/friction_ratio, kT / unit_friction)
    integrator = ActiveBrownianIntegrator(timestep, collision_rate, particleD, particlefriction)
    gpuid = f"{gpuid}"
    dirname = f"./act_ratio_{activity_ratio}_{beta}_{Bfriction}/run{replicate}"
    os.makedirs(dirname,exist_ok=True)
    reporter = HDF5Reporter(
        folder=dirname,
        max_data_length=1000,
        overwrite=True,
    )
    sim = simulation.Simulation(
        platform="CUDA",
        # for custom integrators, feed a tuple with the integrator class reference and a string specifying type,
        # e.g. "brownian", "variableLangevin", "variableVerlet", or simply "UserDefined" if none of the above.
        integrator=(integrator, "brownian"),
        timestep=timestep,
        temperature=temperature,
        GPU=gpuid,
        collision_rate=collision_rate,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )

    polymer = starting_conformations.grow_cubic(N, int(np.ceil(r)))
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(
        v=np.zeros((N, 3))
    )  # initializes velocities of all monomers to zero (no inertia)
    sim.add_force(forces.spherical_confinement(sim, density=density, k=5.0))
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.1,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=None,
            angle_force_kwargs={},
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": 3.0,  # this will let chains cross sometimes
                # 'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },
            except_bonds=True,
        )
    )
    params = {"beta":beta,"R":25,"l":424}
    sim.add_force(flow(params,199,799))
    tic = time.perf_counter()
    for _ in range(nblocks):  # Do 10 blocks
        sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically.
    toc = time.perf_counter()
    print(f"Ran simulation in {(toc - tic):0.4f}s")
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk


if __name__ == "__main__":
    # Grab task ID and number of tasks
    my_task_id = int(sys.argv[1]) - 1
    num_tasks = int(sys.argv[2])

    # parameters to sweep
    act_values = [1]
    print(act_values)

    # batch to process with this task
    acts_per_task = act_values[my_task_id : len(act_values) : num_tasks] # 1:10:5
    print(acts_per_task)
    for activity_ratio in acts_per_task:
        for i in range(20):
            print(f"Running simulation with activity ratio: {activity_ratio}")
            run_monomer_diffusion(i, 0, N, ids, activity_ratio, beta = 0)
