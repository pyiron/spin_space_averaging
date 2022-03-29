from pyiron_atomistics import Project as PyironProject
import numpy as np
import matplotlib.pylab as plt
from collections import defaultdict
from pyiron_contrib.atomistics.atomistics.job.sqs import SQS


def get_bfgs(s, y, H):
    dH = np.einsum('...i,...j,...->...ij', *2*[y], 1 / np.einsum('...i,...i->...', s, y))
    dH -= np.einsum(
        '...ij,...kl,...j,...l,...->...ik', *2*[H], *2*[s], 1 / np.einsum('...ij,...i,...j->...', H, *2 * [s]),
        optimize=True
    )
    return dH


class Project(PyironProject):
    def __init__(
        self,
        path='',
        user=None,
        sql_query=None,
        default_working_directory=False,
    ):
        super().__init__(
            path=path,
            user=user,
            sql_query=sql_query,
            default_working_directory=default_working_directory,
        )
        self.structure = None
        self.potential = None
        self.n_cores = 80
        self.interpolate_h_mag = True
        self.ready_to_run = True
        self.magmom_manitudes = None
        self._magmoms = None
        self.n_copy = None
        self._symmetry = None

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.structure.get_symmetry()
        return self._symmetry

    def get_relaxed_job(self, structure, pressure=False):
        job = self.create.job.Lammps(('lmp_relax', pressure))
        if job.status.finished:
            return job
        job.structure = structure
        if pressure:
            job.calc_minimize(pressure=[0, 0, 0])
        else:
            job.calc_minimize()
        job.run()
        return job

    @property
    def lmp_hessian(self):
        if self.structure is None:
            raise AssertionError('Structure not set')
        lmp = self.create.job.Lammps(('lmp_qha', self.structure, self.potential))
        lmp.structure = self.structure
        if self.potential is None:
            self.potential = lmp.list_potentials()[0]
        lmp.potential = self.potential
        lmp.interactive_open()
        qha = lmp.create_job('QuasiHarmonicApproximation', lmp.job_name.replace('lmp', 'qha'))
        qha.input['num_points'] = 1
        if qha.status.initialized:
            qha.run()
        return qha['output/force_constants'][0]

    def set_input(self, job, fix_spin=True):
        job.set_convergence_precision(electronic_energy=1e-6)
        job.set_encut(encut=550)
        job.set_kpoints(k_mesh_spacing=0.1)
        job.set_mixing_parameters(density_residual_scaling=0.3, spin_residual_scaling=0.3)
        if fix_spin:
            job.fix_spin_constraint = True
        job.server.cores = self.n_cores
        job.server.queue = 'cm'
        job.calc_static()

    @property
    def magmoms(self):
        if self._magmoms is None:
            raise ValueError('magmoms not set yet - execute run_sqs')
        return self._magmoms

    def run_sqs(self, cutoff, n_copy, n_steps=5000, sigma=0.05, max_sigma=4, n_points=100, min_sample_value=1.0e-8):
        sqs = SQS(
            structure=self.structure,
            concentration=0.5,
            cutoff=cutoff,
            n_copy=n_copy,
            sigma=sigma,
            max_sigma=max_sigma,
            n_points=n_points,
            min_sample_value=min_sample_value
        )
        self.n_copy = n_copy
        sqs.run_mc(n_steps)
        self._magmoms = sqs.spins

    def run_init_magmoms(self):
        if self.magmom_magnitudes is None:
            raise ValueError('magmom magnitudes not defined')
        for mag in self.magmom_magnitudes:
            for ii, mm in enumerate(self.magmoms):
                job = pr.create.job.Sphinx(('spx_v', self.structure, mag, ii, 0))
                job.structure = structure
                job.structure.set_initial_magnetic_moments(mag * mm)
                self.set_input(job)
                job.run()

    def get_output(self, job_list, pr=None, shape=None):
        if pr is None:
            pr = self
        output = defaultdict(list)
        for job_name in job_list:
            job = pr.load(job_name)
            output['energy'].append(job.output.energy_pot[-1])
            output['ediff'].append(np.diff(job['output/generic/dft/scf_energy_free'][0])[-1])
            output['nu'].append(job['output/generic/dft/magnetic_forces'][0])
            output['magmoms'].append(job['output/generic/dft/atom_spins'][0])
            output['forces'].append(job['output/generic/forces'][0])
            output['positions'].append(job['output/generic/positions'][0])
        if shape is not None:
            output['magmoms'] = np.array(output['magmoms']).reshape(shape + (-1,))
            output['nu'] = np.array(output['nu']).reshape(shape + (-1,))
            output['forces'] = np.array(output['forces']).reshape(shape + (-1, 3,))
            output['positions'] = np.array(output['positions']).reshape(shape + (-1, 3,))
        return output

    def get_init_hessian_mag(self, pr=None):
        job_lst = [
            ('spx_v', self.structure, mag, ii, 0)
            for mag in self.magmom_magnitudes
            for ii in range(self.n_copy)
        ]
        output = self.get_output(
            job_lst, pr=pr, shape=(len(self.magmom_magnitudes), self.n_copy)
        )
        self.set_initial_H_mag(output['nu'], output['magmoms'])

    def update_hessian(self, magnetic_forces, magmoms, positions, forces, n_cycle):
        nu = np.mean([
            nu[self.symmetry.permutations]
            for nu in np.mean(magnetic_forces * np.sign(magmoms), axis=1)
        ], axis=1)
        f_sym = np.array([
            self.symmetry.symmetrize_vectors(f)
            for f in forces.mean(axis=1)
        ]).reshape(n_cycle, -1)
        x_diff = np.diff(positions, axis=0)
        x_diff = self.structure.find_mic(x_diff).reshape(n_cycle - 1, -1)
        x_diff = np.append(
            x_diff, np.diff(np.absolute(magmoms).mean(axis=1), axis=0), axis=1
        )
        dUdx = np.append(-f_sym, nu, axis=1)
        for xx, ff in zip(x_diff, np.diff(dUdx, axis=0)):
            self.H_current += get_bfgs(xx, ff, self.H_current)

    def set_initial_H_mag(self, magnetic_forces=None, magmoms=None, hessian=None):
        if magnetic_forces is not None and magmoms is not None:
            self.H_mag_init = np.squeeze(
                np.diff(magnetic_forces, axis=0) / np.diff(magmoms, axis=0)
            )
            self.H_mag_init = np.mean(
                np.mean(self.H_mag_init, axis=0)[self.symmetry.permutations], axis=0
            )
        elif hessian is not None:
            self.H_mag_init = hessian
        else:
            raise ValueError('input values not set')
        self.H_mag_init = np.eye(len(self.H_mag_init)) * self.H_mag_init
        if len(self.H_mag_init) != len(self.structure):
            raise AssertionError('Length not correct')
        self.set_initial_H(self.lmp_hessian, self.H_mag_init)

    def get_dx(self, forces, magnetic_forces, magmoms=None, symmetrize=False):
        if symmetrize:
            if magmoms is None:
                raise ValueError('when symmetrize is on magmoms is required')
            magnetic_forces = np.mean([
                nu[self.symmetry.permutations]
                for nu in np.mean(magnetic_forces * np.sign(magmoms), axis=1)
            ], axis=1)
            forces = self.symmetry.symmetrize_vectors(forces.mean(axis=0))
        xm_new = np.einsum('ij,j->i', np.linalg.inv(self.H_current), np.append(-forces, nu[-1]))
        dx = -xm_new[:3 * len(self.structure)].reshape(-1, 3)
        dm = -xm_new[3 * len(self.structure):]
        return dx, dm

    def set_initial_H(self, H_phonon, H_magnon):
        n = len(self.structure)
        self.H_init = np.eye(4 * n)
        self.H_init[:3 * n, :3 * n] = H_phonon.copy()
        self.H_init[3 * n:, 3 * n:] *= H_magnon
        self.H_current = self.H_init.copy()
