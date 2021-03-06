from pyiron_atomistics import Project as PyironProject
import numpy as np
from collections import defaultdict
from spin_space_averaging.sqs import SQSInteractive
from pyiron_contrib.atomistics.atomistics.master.qha import QuasiHarmonicApproximation


def get_bfgs(s, y, H):
    dH = np.einsum('...i,...j,...->...ij', *2 * [y], 1 / np.einsum('...i,...i->...', s, y))
    dH -= np.einsum(
        '...ij,...kl,...j,...l,...->...ik', *2 * [H], *2 * [s], 1 / np.einsum('...ij,...i,...j->...', H, *2 * [s]),
        optimize=True
    )
    return dH


class Project(PyironProject):
    """
    Welcome to the Spin Space Average workflow
    """

    def get_lmp_relaxed_structure(
        self, structure, potential=None, symmetry=None, pressure=None, delete_existing_job=False
    ):
        job = self.create.job.Lammps(
            ('lmp_relax', structure), delete_existing_job=delete_existing_job
        )
        if symmetry is None:
            symmetry = structure.get_symmetry()
        job.structure = structure
        if potential is not None:
            job.potential = potential
        job.calc_minimize(pressure=pressure)
        if job.status.initialized:
            job.run()
        structure.positions += symmetry.symmetrize_vectors(
            job.output.total_displacements[-1]
        )
        return structure.center_coordinates_in_unit_cell()

    def get_lmp_hessian(self, structure, potential=None):
        lmp = self.create.job.Lammps(('lmp', structure))
        lmp.structure = structure
        if potential is not None:
            lmp.potential = potential
        lmp.interactive_open()
        qha = self.create_job(QuasiHarmonicApproximation, 'qha_' + lmp.job_name)
        qha.ref_job = lmp
        qha.input['num_points'] = 1
        if qha.status.initialized:
            qha.run()
        return qha['output/force_constants'][0]

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
            job.structure.spin_constraint[job.structure.select_index('C')] = False
        job.server.cores = n_cores
        job.server.queue = 'cm'
        job.calc_static()

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
        sqs = SQSInteractive(
            structure=structure[indices],
            concentration=0.5,
            cutoff=cutoff,
            n_copy=n_copy,
            sigma=sigma,
            max_sigma=max_sigma,
            n_points=n_points,
            min_sample_value=min_sample_value
        )
        sqs.run_mc(n_steps)
        magmoms = np.zeros((n_copy, len(structure)))
        magmoms.T[indices] = sqs.spins.T
        return magmoms

    def get_output(self, job_list, pr=None, shape=None):
        if pr is None:
            pr = len(job_list) * [self]
        if not isinstance(pr, list):
            pr = len(job_list) * [pr]
        output = defaultdict(list)
        for job_name, prr in zip(job_list, pr):
            job = prr.load(job_name)
            output['energy'].append(job.output.energy_pot[-1])
            output['ediff'].append(np.diff(job['output/generic/dft/scf_energy_free'][0])[-1])
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

    def symmetrize_magmoms(self, symmetry, magmoms, signs=None):
        if signs is None:
            signs = np.sign(magmoms)
        signs = np.sign(signs)
        magmoms = np.atleast_3d(np.asarray(magmoms).T).T
        signs = np.atleast_3d(signs.T).T
        return np.mean([
            mm[symmetry.permutations] for mm in np.mean(magmoms * signs, axis=1)
        ], axis=1).squeeze()

    def update_hessian(
        self, structure, hessian, magnetic_forces, magmoms, positions, forces, symmetry=None
    ):
        if symmetry is None:
            symmetry = structure.get_symmetry()
        nu = self.symmetrize_magmoms(symmetry, magnetic_forces, magmoms)
        magmoms = self.symmetrize_magmoms(symmetry, magmoms)
        f_sym = symmetry.symmetrize_vectors(forces.mean(axis=1)).reshape(-1, 3 * len(structure))
        x_diff = np.diff(positions, axis=0)
        x_diff = structure.find_mic(x_diff).reshape(-1, 3 * len(structure))
        x_diff = np.append(x_diff, np.diff(magmoms, axis=0), axis=1)
        dUdx = np.append(-f_sym, nu, axis=1)
        new_hessian = hessian.copy()
        for xx, ff in zip(x_diff, np.diff(dUdx, axis=0)):
            new_hessian += get_bfgs(xx, ff, new_hessian)
        return new_hessian

    def get_initial_H_mag(self, magnetic_forces, magmoms, symmetry):
        """
            shape: (m_states, n_copy)
        """
        mm = np.sum(magmoms**2, axis=0)
        mn = np.sum(magmoms * magnetic_forces, axis=0)
        m = np.sum(magmoms, axis=0)
        n = np.sum(magnetic_forces, axis=0)
        H = (len(magmoms) * mn - m * n) / (len(magmoms) * mm - m**2)
        return np.mean(
            np.mean(H, axis=0)[symmetry.permutations], axis=0
        )

    def get_dx(self, hessian, forces, magnetic_forces, symmetry=None, magmoms=None):
        if symmetry is not None:
            if magmoms is not None:
                magnetic_forces = self.symmetrize_magmoms(symmetry, magnetic_forces, magmoms)
            forces = symmetry.symmetrize_vectors(forces.mean(axis=0))
        xm_new = np.einsum('ij,j->i', np.linalg.inv(hessian), np.append(-forces, magnetic_forces))
        dx = -xm_new[:3 * forces.shape[-2]].reshape(-1, 3)
        dm = -xm_new[3 * forces.shape[-2]:]
        return dx, dm

    def get_initial_H(self, H_phonon, H_magnon):
        n = (len(H_phonon) + len(H_magnon)) // 4
        H = np.eye(4 * n)
        H[:3 * n, :3 * n] = H_phonon.copy()
        H[3 * n:, 3 * n:] *= H_magnon
        return H
