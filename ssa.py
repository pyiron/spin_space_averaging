import numpy as np
from collections import defaultdict
from spin_space_averaging.sqs import SQS
from pyiron_contrib.atomistics.atomistics.master.qha import QuasiHarmonicApproximation
from pyiron_base.jobs.job.util import _get_safe_job_name
from pyiron_atomistics.interactive.quasi_newton import run_qn


def get_bfgs(s, y, H):
    dH = np.einsum(
        '...i,...j,...->...ij',
        *2 * [y],
        1 / np.einsum('...i,...i->...', s, y)
    )
    dH -= np.einsum(
        '...ij,...kl,...j,...l,...->...ik',
        *2 * [H],
        *2 * [s],
        1 / np.einsum('...ij,...i,...j->...', H, *2 * [s]),
        optimize=True
    )
    return dH


def get_psb(s, y, H):
    H_tmp = y - np.einsum("...ij,...j->...i", H, s)
    inv_ss = 1 / np.einsum("...i,...i->...", s, s)
    dH = np.einsum("...i,...j,...->...ij", H_tmp, s, inv_ss, optimize=True)
    dH += dH.T
    return (
        dH
        - np.einsum("...i,...i,...j,...k,...->...jk", s, H_tmp, s, s, inv_ss**2, optimize=True)
    )


def get_asym_sum(args):
    return np.sum([
        np.sin(ii + 2) * arg for ii, arg in enumerate(args)
    ])


def struct_to_tag(structure):
    d = structure.get_distances_array(structure.positions, structure.positions)
    indices = np.triu_indices(len(d), 1)
    d = np.log(d[indices])
    c = structure.get_chemical_symbols()[np.stack(indices, axis=-1)]
    d_round = np.round(d, decimals=8)
    Jij = [np.sum([ord(ccc) for ccc in ''.join(np.sort(cc))]) for cc in c]
    E = np.sum(Jij * d_round)
    cell = np.round(structure.cell, decimals=8)
    arr = tuple(structure.pbc) + tuple(cell.flatten()) + (E, )
    return np.log(np.absolute(hash(arr)))


class Lammps:
    def __init__(self, ref_job):
        self._ref_job = ref_job
        self._structure = None

    @property
    def project(self):
        return self._ref_job.project

    def get_hessian(self, structure, potential=None):
        lmp = self.project.create.job.Lammps((
            'lmp', struct_to_tag(structure)
        ))
        lmp.structure = self._get_minimize(
            structure, potential=potential, pressure=np.zeros(3)
        )
        if potential is not None:
            lmp.potential = potential
        lmp.interactive_open()
        qha = self.project.create_job(
            QuasiHarmonicApproximation, 'qha_' + lmp.job_name
        )
        qha.ref_job = lmp
        qha.input['num_points'] = 1
        if qha.status.initialized:
            qha.run()
        return qha['output/force_constants'][0]

    def _get_minimize(
        self,
        structure,
        potential=None,
        pressure=None,
        bulk_modulus=1e3,
        min_change=1.0e-4,
        pressure_tolerance=1.0e-2
    ):
        job_name = ('lmp_relax', struct_to_tag(structure), pressure)
        if pressure is not None:
            job_name = ('lmp_relax', struct_to_tag(structure), *pressure)
        job = self.project.create.job.Lammps(job_name)
        job.structure = structure.copy()
        if potential is not None:
            job.potential = potential
        if job.status.initialized:
            with job.interactive_open() as lmp:
                qn = run_qn(lmp)
                if pressure is not None:
                    H = lmp.structure.get_volume()**(1 / 3) / np.ones((3, 3)) / bulk_modulus
                    for _ in range(100):
                        lmp.structure.set_cell(
                            lmp.structure.cell + lmp.output.pressures[-1] * H, scale_atoms=True
                        )
                        qn = run_qn(lmp, starting_h=qn.hessian)
                        dp = np.prod(lmp.output.pressures[-2:], axis=0)
                        H[dp < -pressure_tolerance**2] *= 0.5
                        H[dp > pressure_tolerance**2] *= 1.2

                        if np.absolute(
                            lmp.output.pressures[-1].diagonal() - pressure
                        ).max() < pressure_tolerance:
                            break
        return job.get_structure()

    @property
    def structure(self):
        if self._structure is None:
            self._structure = self._get_minimize(
                self._ref_job.structure,
                self._ref_job.input.lammps['potential'],
                None
            )
        return self._structure


class SSA:
    """
    Spin-space-averaging workflow - this allows for a paramagnetic structure
    optimization using multiple SQS snapshots.

    This workflow contains the following steps:

    1. Structure pre-optimization using LAMMPS
    2. Computation of SQS
    3. Estimation of the diagonal Hessian matrix for the magnetic moments
    4. Estimation of the Hessian matrix for the positions using LAMMPS
    5. Launch calculations according to the Quasi Newton algorithm

    Detailed explanation and customization possibilities for each step:

    1.
    - The pre-optimization can be switched off by setting
        `job.input.lammps.use_lammps = False`
    - The interatomic potential can be provided via
        `job.input.lammps.potential`, but otherwise the workflow tries to find
        a random potential
    2.
    - The number of snapshots for SQS (and the following simulations) is given
        in `job.input.n_copy`. Default is 8
    - SQS takes only pairwise distribution into account, but the statistics to
        approximate the infinite medium is performed using all `n_copy`
        snapshots. Furthermore, the algorithm relies on the smeared histogram,
        so the smearing width (given in `job.input.sqs.sigma`) should be small
        enough to distinguish different shells, but not too small if the system
        contains structural defects (such as vacancy or grain boundary) in
        order to give good statistics.
    - The indices of the non-magnetic atoms (such as C or H) must be stored in
        `job.input.nonmag_atoms``
    3.
    - In order to initialize the Hessian matrix for the magnetic interactions,
        set either the sample magnetic moments in
        `job,input.init_hessian.magnetic_moments` or set the Hessian matrix
        directly in `self.input.init_hessian.magnon`.
    """
    def __init__(self, project, name):
        self._project = project.create_group(name)
        self.lammps = Lammps(self)
        self._symmetry = None
        self._all_jobs = {}
        self._initial_hessian = None
        try:
            self.project.data.read()
        except KeyError:
            self.input.n_copy = 8
            self.input.is_diffusion = False
            self.input.nonmag_atoms = None
            self.input.create_group('lammps')
            self.input.lammps.potential = None
            self.input.lammps.use_lammps = True
            self.input.lammps.mode = 'PSB'
            self.input.lammps.ionic_steps = 10000
            self.input.lammps.ionic_force_tolerance = 1.0e-4
            self.input.lammps.starting_h = 10
            self.input.lammps.max_displacement = 0.1
            self.input.lammps.bulk_modulus = 1e4
            self.input.create_group('symmetry')
            self.input.symmetry.symprec = 1e-05
            self.input.create_group('sqs')
            self.input.sqs.cutoff = 10
            self.input.sqs.sigma = 0.05
            self.input.sqs.max_sigma = 4
            self.input.sqs.n_points = 200
            self.input.sqs.min_sample_value = 1.0e-8
            self.input.sqs.n_steps = 10000
            self.input.sqs.snapshots = None
            self.input.create_group('init_hessian')
            # Initial total Hessian not defined!!
            self.input.init_hessian.phonon = None
            self.input.init_hessian.magnon = None
            self.input.init_hessian.magnetic_moments = None
            self.input.create_group('dft')
            self.input.dft.electronic_energy = 1e-6
            self.input.dft.encut = 550
            self.input.dft.k_mesh_spacing = 0.1
            self.input.dft.mixing_parameter = 1.0
            self.input.dft.residual_scaling = 0.3
            self.input.dft.n_cores = 80
            self.input.create_group('convergence')
            self.input.convergence.phonon_force = 1.0e-2
            self.input.convergence.magnon_force = 1.0e-1
            self.input.convergence.max_steps = 100
            self.sync()

    @property
    def _dft_job_name(self):
        dft = self.input.dft
        return get_asym_sum(
            [dft[k] for k in sorted(dft.list_nodes()) if k != 'n_cores']
        )

    @property
    def _structure_job_name(self):
        return struct_to_tag(self.structure)

    def set_nonmag_atoms(self, ids):
        self.input.nonmag_atoms = ids
        self.sync()

    @property
    def structure(self):
        try:
            return self.input.structure
        except AttributeError:
            raise AssertionError('structure not defined')

    @structure.setter
    def structure(self, new_structure):
        self.input.structure = new_structure
        self.sync()

    @property
    def _relaxed_structure(self):
        try:  # backward compatibility
            if not self.input.lammps.use_lammps:
                return self.structure
        except KeyError:
            pass
        return self.lammps.structure

    def get_symmetry(self, structure, symprec):
        return structure.get_symmetry(symprec=symprec)

    def get_sqs(
        self,
        structure,
        cutoff,
        n_copy,
        nonmag_ids=None,
        n_steps=5000,
        sigma=0.05,
        max_sigma=4,
        n_points=100,
        min_sample_value=1.0e-8
    ):
        job_name = _get_safe_job_name((
            'sqs',
            self._structure_job_name,
        ))
        sqs = self.project.create_job(SQS, job_name)
        sqs.structure = structure
        sqs.input.concentration = 0.5
        sqs.input.cutoff = cutoff
        sqs.input.n_copy = n_copy
        sqs.input.sigma = sigma
        sqs.input.max_sigma = max_sigma
        sqs.input.n_points = n_points
        sqs.input.min_sample_value = min_sample_value
        if sqs.status.initialized:
            sqs.run()
        magmoms = np.zeros((n_copy, len(structure)))
        magmoms = sqs.output.spins
        if nonmag_ids is not None:
            magmoms[:, nonmag_ids] *= 0
        return magmoms

    @property
    def sqs(self):
        if self.input.sqs.snapshots is None:
            self.input.sqs.snapshots = self.get_sqs(
                self._relaxed_structure,
                self.input.sqs.cutoff,
                self.input.n_copy,
                nonmag_ids=self.input.nonmag_atoms,
                n_steps=self.input.sqs.n_steps,
                sigma=self.input.sqs.sigma,
                max_sigma=self.input.sqs.max_sigma,
                n_points=self.input.sqs.n_points,
                min_sample_value=self.input.sqs.min_sample_value
            )
            self.sync()
        return self.input.sqs.snapshots

    def set_input(
        self,
        job,
        fix_spin=True,
        electronic_energy=1e-6,
        encut=550,
        k_mesh_spacing=0.1,
        mixing_parameter=1.0,
        residual_scaling=0.3,
        n_cores=80,
    ):
        job.set_convergence_precision(electronic_energy=electronic_energy)
        job.set_encut(encut=encut)
        job.set_kpoints(k_mesh_spacing=k_mesh_spacing)
        job.set_mixing_parameters(
            density_residual_scaling=residual_scaling,
            density_mixing_parameter=mixing_parameter,
            spin_residual_scaling=residual_scaling,
            spin_mixing_parameter=mixing_parameter,
        )
        if fix_spin:
            job.fix_spin_constraint = True
            if self.input.nonmag_atoms is not None:
                job.structure.spin_constraint[self.input.nonmag_atoms] = False
        job.server.cores = n_cores
        job.server.queue = 'cm'
        job.calc_static()

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.get_symmetry(
                self.structure, self.input.symmetry.symprec
            )
        return self._symmetry

    @property
    def init_magmom_jobs(self):
        if self.input.init_hessian.magnetic_moments is None:
            raise ValueError(
                'job.input.init_hessian.magnetic_moments not defined'
            )
        if len(self.input.init_hessian.magnetic_moments) <= 1:
            raise ValueError(
                'There must be at least two magnetic moment values'
            )
        is_running = False
        job_lst = []
        for magnitude in np.atleast_1d(self.input.init_hessian.magnetic_moments):
            for j, magmoms in enumerate(self.sqs):
                job_name = (
                    'spx',
                    self._structure_job_name,
                    self._dft_job_name,
                    magnitude,
                    get_asym_sum(self.sqs.flatten()),
                    j,
                )
                if self.get_job(job_name) is None:
                    spx = self.project.create.job.Sphinx(job_name)
                    spx.structure = self._relaxed_structure.copy()
                    spx.structure.set_initial_magnetic_moments(magnitude * magmoms)
                    self.set_input(spx)
                    if spx.status.initialized:
                        spx.run()
                    is_running = True
                job_lst.append(job_name)
        if is_running:
            return None
        return job_lst

    @property
    def initial_hessian_phonon(self):
        if self.input.init_hessian.phonon is not None:
            return self.input.init_hessian.phonon
        return self.lammps.get_hessian(
            self.structure,
            self.input.lammps.potential
        )

    @property
    def initial_hessian_magnon(self):
        if self.input.init_hessian['magnon'] is None:
            if self.input.init_hessian.magnetic_moments is None:
                raise ValueError(
                    'job.input.init_hessian.magnetic_moments not defined'
                )
            job_lst = self.init_magmom_jobs
            if job_lst is None:
                return
            output = self.get_job_output(
                job_lst, (
                    len(self.input.init_hessian.magnetic_moments),
                    len(self.sqs)
                )
            )
            self.input.init_hessian['magnon'] = self.get_initial_hessian_magnon(
                self.symmetrize_magmoms(
                    self.symmetry, output['nu'], output['magmoms']
                ),
                self.symmetrize_magmoms(self.symmetry, output['magmoms']),
                self.symmetry,
            )
            if np.any(self.input.init_hessian['magnon'] <= 0):
                raise ValueError('Negative curvature detected:', self.input.init_hessian['magnon'])
            self.sync()
        return self.input.init_hessian['magnon']

    @property
    def initial_hessian(self):
        if self._initial_hessian is None:
            self._initial_hessian = self.get_initial_hessian(
                H_phonon=self.initial_hessian_phonon,
                H_magnon=self.initial_hessian_magnon
            )
        return self._initial_hessian.copy()

    def get_initial_hessian(self, H_phonon, H_magnon):
        if H_phonon is None or H_magnon is None:
            return None
        n = (len(H_phonon) + len(H_magnon)) // 4
        H = np.eye(4 * n)
        H[:3 * n, :3 * n] = H_phonon.copy()
        H[3 * n:, 3 * n:] *= H_magnon
        return H

    def get_initial_hessian_magnon(
        self, magnetic_forces, magmoms, symmetry, weights=None
    ):
        """
        shape: (m_states, n_copy)
        """
        if weights is None:
            weights = np.ones(len(magmoms))
        mm = np.einsum('ij,i->j', magmoms**2, weights)
        mn = np.einsum('ij,ij,i->j', magmoms, magnetic_forces, weights)
        m = np.einsum('ij,i->j', magmoms, weights)
        n = np.einsum('ij,i->j', magnetic_forces, weights)
        w = np.sum(weights)
        H = (w * mn - m * n) / (w * mm - m**2)
        if self.input.nonmag_atoms is not None:
            # Any positive number to make sure that the inverse matrix is well defined
            H[self.input.nonmag_atoms] = 1
        return np.mean(H[symmetry.permutations], axis=0)

    def symmetrize_magmoms(self, symmetry, magmoms, signs=None):
        if signs is None:
            signs = np.sign(magmoms)
        signs = np.sign(signs)
        magmoms = np.atleast_3d(np.asarray(magmoms).T).T
        signs = np.atleast_3d(signs.T).T
        return np.mean([
            mm[symmetry.permutations] for mm in np.mean(magmoms * signs, axis=1)
        ], axis=1).squeeze()

    def get_job(self, job_name):
        job_name = _get_safe_job_name(job_name)
        try:
            return self._all_jobs[job_name]
        except KeyError:
            jt = self.project.job_table()
            if job_name not in list(jt.job):
                return
            if job_name in (
                list(jt[jt.status == 'submitted'].job)
                + list(jt[jt.status == 'running'].job)
            ):
                return
            self._all_jobs[job_name] = self.project.load(job_name)
            return self._all_jobs[job_name]

    def get_job_output(self, job_list, shape=None):
        output = defaultdict(list)
        assert shape is None or isinstance(shape, tuple)
        for job_name in job_list:
            job = self.get_job(job_name)
            output['energy'].append(job.output.energy_pot[-1])
            output['ediff'].append(
                np.diff(job['output/generic/dft/scf_energy_free'][-1])[-1]
            )
            output['nu'].append(job['output/generic/dft/magnetic_forces'][-1])
            output['magmoms'].append(job['output/generic/dft/atom_spins'][-1])
            output['forces'].append(job['output/generic/forces'][-1])
            output['positions'].append(job['output/generic/positions'][-1])
            output['residue'].append(job['output/generic/dft/residue'][-1])
        if shape is not None:
            output['energy'] = np.reshape(output['energy'], shape)
            shape = output['energy'].shape
            output['ediff'] = np.reshape(output['ediff'], shape)
            output['residue'] = np.reshape(output['residue'], shape + (2,))
            output['magmoms'] = np.reshape(output['magmoms'], shape + (-1,))
            output['nu'] = np.reshape(output['nu'], shape + (-1,))
            output['forces'] = np.reshape(output['forces'], shape + (-1, 3,))
            output['positions'] = np.reshape(output['positions'], shape + (-1, 3,))
        return output

    @property
    def project(self):
        return self._project

    @property
    def input(self):
        if 'input' not in self.project.data:
            self.project.data.create_group('input')
        return self.project.data.input

    def sync(self):
        self.project.data.write()

    def update_hessian(
        self,
        structure,
        hessian,
        magnetic_forces,
        magmoms,
        positions,
        forces,
        symmetry=None
    ):
        if len(magmoms) == 1:
            return np.array([hessian])
        if symmetry is None:
            symmetry = structure.get_symmetry()
        nu = self.symmetrize_magmoms(symmetry, magnetic_forces, magmoms)
        magmoms = self.symmetrize_magmoms(symmetry, magmoms)
        f_sym = symmetry.symmetrize_vectors(
            forces.mean(axis=1)
        ).reshape(-1, 3 * len(structure))
        # assert np.shape(positions) == f_sym.shape
        x_diff = np.diff(positions, axis=0)
        x_diff = structure.find_mic(x_diff).reshape(-1, 3 * len(structure))
        x_diff = np.append(x_diff, np.diff(magmoms, axis=0), axis=1)
        dUdx = np.append(-f_sym, nu, axis=1)
        new_hessian = [hessian.copy()]
        for xx, ff in zip(x_diff, np.diff(dUdx, axis=0)):
            if self.is_diffusion:
                new_hessian.append(new_hessian[-1] + get_psb(xx, ff, new_hessian[-1]))
            else:
                new_hessian.append(new_hessian[-1] + get_bfgs(xx, ff, new_hessian[-1]))
        return np.asarray(new_hessian)

    @property
    def is_diffusion(self):
        try:
            return self.input.is_diffusion
        except KeyError:
            return False

    def _get_dx(self, hessian, forces, magnetic_forces, symmetry=None, magmoms=None):
        if symmetry is not None:
            if magmoms is not None:
                magnetic_forces = np.array([
                    self.symmetrize_magmoms(symmetry, nu, m)
                    for nu, m in zip(magnetic_forces, magmoms)
                ])
            forces = np.array([symmetry.symmetrize_vectors(f.mean(axis=0)) for f in forces])
        # assert len(hessian) == len(forces) == len(magnetic_forces)
        # assert np.ptp(np.shape(hessian)[-2:]) == 0
        dx, dm = [], []
        for h, f, m in zip(hessian, forces, magnetic_forces):
            xm_new = np.einsum('ij,j->i', np.linalg.inv(h), np.append(-f, m))
            dx.append(-xm_new[:3 * forces.shape[-2]].reshape(-1, 3))
            dm.append(-xm_new[3 * forces.shape[-2]:])
        return np.asarray(dx), np.asarray(dm)

    def _check_convergence(self, f, nu, m):
        f_sym = self.symmetry.symmetrize_vectors(f.mean(axis=0))
        if np.linalg.norm(
            f_sym, axis=-1
        ).max() > self.input.convergence.phonon_force:
            return False
        nu_sym = self.symmetrize_magmoms(self.symmetry, nu, m)
        return np.absolute(nu_sym).max() < self.input.convergence.magnon_force

    def _run_next(self, job_lst, output):
        if self._check_convergence(
            output.all_forces[-1], output.nu[-1], output.magmoms[-1]
        ):
            return
        for ii, job_name in enumerate(job_lst[-len(self.sqs):]):
            spx_old = self.get_job(job_name)
            new_job_name = spx_old.job_name.split('_')
            try:
                new_job_name[-2] = str(int(new_job_name[-2]) + 1)
                new_job_name = '_'.join(new_job_name)
            except:
                new_job_name = (
                    'spx',
                    self._structure_job_name,
                    self._dft_job_name,
                    get_asym_sum(self.sqs.flatten()),
                    0,
                    ii,
                )
            if self.get_job(new_job_name) is not None: continue
            spx = self.project.create.job.Sphinx(new_job_name)
            if not spx.status.initialized: continue
            spx.structure = spx_old.structure.copy()
            spx.structure.positions += output.dx[-1]
            m = spx_old.structure.get_initial_magnetic_moments()
            spx.structure.set_initial_magnetic_moments(m + np.sign(m) * output.dm[-1])
            self.set_input(spx)
            spx.run()

    @property
    def _qn_job_lst(self):
        job_lst = []
        if self.input.init_hessian.magnetic_moments is None:
            if self.input.init_hessian.magnon is None:
                raise AssertionError(
                    'Either magnon Hessian or magmoms must be defined'
                )
        else:
            magmom_jobs = self.init_magmom_jobs
            if magmom_jobs is None: return None
            m = self.input.init_hessian.magnetic_moments
            output = self.get_job_output(magmom_jobs, (-1, len(m)))
            i = np.argmin(output['energy'].mean(axis=0))
            job_lst = magmom_jobs[i * len(m):i * len(m) + len(self.sqs)]
        for i in range(self.input.convergence.max_steps):
            job_lst_tmp = []
            for j, magmoms in enumerate(self.sqs):
                job_tmp = self.get_job((
                    'spx',
                    self._structure_job_name,
                    self._dft_job_name,
                    get_asym_sum(self.sqs.flatten()),
                    i,
                    j,
                ))
                if job_tmp is None: break
                if job_tmp.status not in ['finished', 'not_converged']: break
                job_lst_tmp.append(job_tmp.job_name)
            if len(job_lst_tmp) == len(self.sqs):
                job_lst.extend(job_lst_tmp)
            else: break
        return job_lst

    def get_output(self):
        job_lst = self._qn_job_lst
        if job_lst is None: return None
        job_output = self.get_job_output(job_lst, (-1, len(self.sqs)))
        output = Output(self, job_output)
        self._run_next(job_lst, output)
        return output


class Output:
    def __init__(self, ref_job, all_output):
        self._job = ref_job
        self._all_output = all_output
        self._hessian = None
        self._dx = None
        self._dm = None

    @property
    def all_energy(self):
        if self._all_output is None: return None
        return self._all_output['energy']

    @property
    def energy(self):
        if self._all_output is None: return None
        return np.mean(self.all_energy, axis=-1)

    @property
    def all_forces(self):
        if self._all_output is None: return None
        return self._all_output['forces']

    @property
    def e_diff(self):
        if self._all_output is None: return None
        return self._all_output['ediff']

    @property
    def forces(self):
        if self._all_output is None: return None
        return np.array([
            self._job.symmetry.symmetrize_vectors(f.mean(axis=0))
            for f in self.all_forces
        ])

    @property
    def nu(self):
        if self._all_output is None: return None
        return self._all_output['nu']

    @property
    def magmoms(self):
        if self._all_output is None: return None
        return self._all_output['magmoms']

    @property
    def hessian(self):
        if self._all_output is None: return None
        if self._hessian is None:
            self._hessian = self._job.update_hessian(
                self._job.structure,
                self._job.initial_hessian,
                self._all_output['nu'],
                self._all_output['magmoms'],
                self._all_output['positions'][:, 0],
                self._all_output['forces'],
                self._job.symmetry
            )
        return self._hessian

    @property
    def dx(self):
        if self._all_output is None: return None
        if self._dx is None:
            self._dx, self._dm = self._job._get_dx(
                self.hessian,
                self.all_forces,
                self._all_output['nu'],
                self._job.symmetry,
                self._all_output['magmoms']
            )
        return self._dx

    @property
    def dm(self):
        if self._all_output is None: return None
        if self._dm is None:
            _ = self.dx
        return self._dm

    @property
    def dE(self):
        if self._all_output is None: return None
        E_lst = []
        for dx, dm, h in zip(self.dx, self.dm, self.hessian):
            dx = np.append(dx.flatten(), dm)
            E_lst.append(h.dot(dx).dot(dx))
        return np.array(E_lst)
