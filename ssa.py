from pyiron_atomistics import Project as PyironProject
import numpy as np
from collections import defaultdict
from spin_space_averaging.sqs import SQS
from pyiron_contrib.atomistics.atomistics.master.qha import QuasiHarmonicApproximation
from pyiron_base.job.util import _get_safe_job_name


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


def get_asym_sum(*args):
    return np.sum([
        np.sin(ii + 2) * arg for ii, arg in enumerate(args)
    ])


class SSA:
    def __init__(self, project, name):
        self._project = project.create_group(name)
        self._output = Output(self)
        self._lmp_structure_zero = None
        self._lmp_structure = None
        self._symmetry = None
        self.all_jobs = {}
        self._initial_hessian = None
        self._lmp_hessian = None
        try:
            self.project.data.read()
        except KeyError:
            self.input.n_copy = 8
            self.input.structure = None
            self.input.nonmag_atoms = None
            self.input.create_group('lammps')
            self.input.lammps.potential = None
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
            self.sync()

    def _dft_job_name(self):
        dft = self.input.dft
        return get_asym_sum(
            [dft[k] for k in dft.list_nodes() if k != 'n_cores']
        )

    def _structure_job_name(self):
        box = self.structure
        results = get_asym_sum(*box.positions.flatten())
        results += get_asym_sum(*box.cell.flatten())
        results += get_asym_sum(
            *[ord(s) for s in ''.join(box.get_chemical_symbols())]
        )
        return results        

    def set_nonmag_atoms(self, ids):
        self.input.nonmag_atoms = ids
        self.sync()

    @property
    def structure(self):
        if self.input['structure'] is None:
            raise AssertionError('Structure not set')
        return self.input['structure']

    @structure.setter
    def structure(self, new_structure):
        self.input['structure'] = structure
        self.sync()

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
        indices = np.arange(len(structure))
        if nonmag_ids is not None:
            indices = np.delete(indices, nonmag_ids)
        job_name = _get_safe_job_name((
            'sqs',
            self._structure_job_name,
        ))
        sqs = self.project.create_job(SQS, job_name)
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
        magmoms.T[indices] = sqs.output.spins.T
        return magmoms

    @property
    def sqs(self):
        if self.input.sqs.snapshots is None:
            self.input.sqs.snapshots = self.get_sqs(
                self.structure,
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

    def lmp_hessian(self):
        if self.input.init_hessian.phonon is not None:
            return self.input.init_hessian.phonon
        if self._lmp_hessian is None:
            self._lmp_hessian = self.get_lmp_hessian(
                self.structure, self.input.lammps.potential
            )
        return self._lmp_hessian

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.get_symmetry(
                self.structure, self.input.symmetry.symprec
            )
        return self._symmetry

    def get_lmp_hessian(self, structure, potential=None):
        lmp = self.project.create.job.Lammps(('lmp', self._structure_job_name))
        lmp.structure = structure
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

    def _get_lmp_minimize(
        self, structure, potential=None, symmetry=None, pressure=None
    ):
        job_name = ('lmp_relax', self._structure_job_name, pressure)
        if pressure is not None:
            job_name = ('lmp_relax', self._structure_job_name, *pressure)
        job = self.project.create.job.Lammps(job_name)
        if symmetry is None:
            symmetry = structure.get_symmetry()
        job.structure = structure
        if potential is not None:
            job.potential = potential
        job.calc_minimize(pressure=pressure)
        if job.status.initialized:
            job.run()
        final_structure = job.get_structure()
        dx = final_structure.get_scaled_positions() - structure.get_scaled_positions()
        dx -= np.rint(dx)
        dx = np.einsum('ji,nj->ni', final_structure.cell, dx)
        structure = structure.apply_strain(
            np.einsum(
                'ij,jk->ik',
                final_structure.cell,
                np.linalg.inv(structure.cell)
            ) - np.eye(3)
        )
        structure.positions += symmetry.symmetrize_vectors(dx)
        return structure.center_coordinates_in_unit_cell()

    @property
    def lmp_structure(self):
        if self._lmp_structure is None:
            self._lmp_structure = self._get_lmp_minimize(
                self.structure,
                self.input.lammps['potential'],
                self.symmetry,
                None
            )
        return self._lmp_structure

    @property
    def init_magmom_jobs(self):
        if self.input.init_hessian.magnetic_moments is None:
            raise ValueError(
                'job.input.init_hessian.magnetic_moments not defined'
            )
        is_running = False
        job_lst = []
        for magnitude in self.atleast_1d(self.input.init_hessian.magnetic_moments):
            for magmoms in self.sqs:
                job_name = _get_safe_job_name((
                    'spx',
                    self._structure_job_name,
                    self._dft_job_name,
                    get_asym_sum(magmoms),
                    magnitude
                ))
                spx = self.get_job(job_name)
                if spx is None:
                    spx = self.project.create.job.Sphinx(job_name)
                    spx.structure = self.lmp_structure.copy()
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
    def initial_hessian_mag(self):
        if self.input.init_hessian['magnon'] is None:
            if self.input.init_hessian.magnetic_moments is None:
                raise ValueError(
                    'job.input.init_hessian.magnetic_moments not defined'
                )
            job_lst = self.init_magmom_jobs
            if job_lst is None:
                return
            output = self.get_output(
                job_lst, (len(self.input.init_hessian.magnetic_moments), -1)
            )
            self.input.init_hessian['magnon'] = self.get_initial_hessian_mag(
                output['nu'],
                output['magmoms'],
                self.symmetry,
            )
            self.sync()
        return self.input.init_hessian['magnon']

    @property
    def initial_hessian(self):
        if self._initial_hessian is None:
            H_phonon = self.input_hessian.phonon
            H_magnon = self.initial_hessian_mag
            if H_phonon is None or H_magnon is None:
                return None
            self._initial_hessian = self.get_initial_hessian(
                H_phonon=H_phonon, H_magnon=H_magnon
            )
        return self._initial_hessian

    def get_initial_hessian(self, H_phonon, H_magnon):
        n = (len(H_phonon) + len(H_magnon)) // 4
        H = np.eye(4 * n)
        H[:3 * n, :3 * n] = H_phonon.copy()
        H[3 * n:, 3 * n:] *= H_magnon
        return H

    def get_initial_hessian_mag(
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
        return np.mean(
            np.mean(H, axis=0)[symmetry.permutations], axis=0
        )

    @property
    def lmp_structure_zero(self):
        if self._lmp_structure_zero is None:
            self._lmp_structure_zero = self._get_lmp_minimize(
                self.structure,
                self.input.lammps['potential'],
                self.symmetry,
                [0, 0, 0]
            )
        return self._lmp_structure_zero

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
        if job_name not in list(self.all_job.keys()):
            if job_name not in list(self.project.job_table().jobs):
                return
            job = self.project.load(job_name)
            if job.status.running:
                return None
            self.all_job[job_name] = job
        return self.all_job[job_name]

    def get_output(self, job_list, shape=None):
        output = defaultdict(list)
        for job_name in job_list:
            job = self.get_job(job_name)
            output['energy'].append(job.output.energy_pot[-1])
            output['ediff'].append(
                np.diff(job['output/generic/dft/scf_energy_free'][0])[-1]
            )
            output['nu'].append(job['output/generic/dft/magnetic_forces'][0])
            output['magmoms'].append(job['output/generic/dft/atom_spins'][0])
            output['forces'].append(job['output/generic/forces'][0])
            output['positions'].append(job['output/generic/positions'][0])
        if shape is not None:
            output['energy'] = np.reshape(output['energy'], shape)
            output['ediff'] = np.reshape(output['ediff'], shape)
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

    @property
    def output(self):
        return self._output

    def sync(self):
        self.project.data.write()


class Output:
    def __init__(self, ref_job):
        self._job = ref_job


class Project(PyironProject):
    """
    Welcome to the Spin Space Average workflow
    """

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
        if symmetry is None:
            symmetry = structure.get_symmetry()
        nu = self.symmetrize_magmoms(symmetry, magnetic_forces, magmoms)
        magmoms = self.symmetrize_magmoms(symmetry, magmoms)
        f_sym = symmetry.symmetrize_vectors(
            forces.mean(axis=1)
        ).reshape(-1, 3 * len(structure))
        x_diff = np.diff(positions, axis=0)
        x_diff = structure.find_mic(x_diff).reshape(-1, 3 * len(structure))
        x_diff = np.append(x_diff, np.diff(magmoms, axis=0), axis=1)
        dUdx = np.append(-f_sym, nu, axis=1)
        new_hessian = hessian.copy()
        for xx, ff in zip(x_diff, np.diff(dUdx, axis=0)):
            new_hessian += get_bfgs(xx, ff, new_hessian)
        return new_hessian

    def get_dx(self, hessian, forces, magnetic_forces, symmetry=None, magmoms=None):
        if symmetry is not None:
            if magmoms is not None:
                magnetic_forces = self.symmetrize_magmoms(
                    symmetry, magnetic_forces, magmoms
                )
            forces = symmetry.symmetrize_vectors(forces.mean(axis=0))
        xm_new = np.einsum(
            'ij,j->i', np.linalg.inv(hessian), np.append(-forces, magnetic_forces)
        )
        dx = -xm_new[:3 * forces.shape[-2]].reshape(-1, 3)
        dm = -xm_new[3 * forces.shape[-2]:]
        return dx, dm
