import numpy as np
from tqdm.auto import tqdm
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pyiron_base.generic.genericinput import GenericInput, InputField


class SQSInteractive:
    def __init__(
        self,
        structure,
        concentration,
        cutoff,
        n_copy=1,
        sigma=0.05,
        max_sigma=4,
        n_points=200,
        min_sample_value=1.0e-8
    ):
        self.structure = structure
        self.concentration = concentration
        self.cutoff = cutoff
        self.sigma = sigma
        self.n_points = n_points
        self.max_sigma = max_sigma
        self._neigh = None
        self._indices = None
        self._x_range = None
        self._prefactor = None
        self._kappa = None
        self._histogram = None
        self._spins = None
        self._cond = None
        self._n_copy = n_copy
        self.min_sample_value = min_sample_value
        self._sample_points = None

    @property
    def n_copy(self):
        return self._n_copy

    @n_copy.setter
    def n_copy(self, value):
        self._spins = None
        self._n_copy = value

    @property
    def neigh(self):
        if self._neigh is None:
            self._neigh = self.structure.get_neighbors(
                num_neighbors=None, cutoff_radius=self.cutoff
            )
        return self._neigh

    @property
    def indices(self):
        if self._indices is None:
            self._indices = np.stack((
                self.neigh.flattened.indices, self.neigh.flattened.atom_numbers
            ), axis=-1)
            self._cond = np.diff(self._indices, axis=-1).flatten() > 0
            self._indices = self._indices[self._cond]
        return self._indices

    @property
    def cond(self):
        if self._cond is None:
            _ = self.indices
        return self._cond

    @property
    def x_range(self):
        if self._x_range is None:
            self._x_range = np.linspace(
                self.neigh.flattened.distances.min() - self.max_sigma * self.sigma,
                self.neigh.flattened.distances.max() + self.max_sigma * self.sigma,
                self.n_points
            )
        return self._x_range

    @property
    def prefactor(self):
        if self._prefactor is None:
            self._prefactor = 1 / np.sqrt(2 * np.pi * self.sigma**2) / np.sum(self.cond)
        return self._prefactor

    @property
    def kappa(self):
        if self._kappa is None:
            x_diff = self.x_range[:, None] - self.neigh.flattened.distances[self.cond]
            self._kappa = np.exp(
                -x_diff**2 / (2 * self.sigma**2)
            ) / self.x_range[:, None]**2
        return self._kappa

    @property
    def histogram(self):
        if self._histogram is None:
            self._histogram = self.kappa.sum(axis=1) * self.prefactor
        return self._histogram

    @property
    def spins(self):
        if self._spins is None:
            self._spins = np.ones(len(self.structure) * self.n_copy).astype(int)
            self._spins[:np.rint(len(self._spins) * self.concentration).astype(int)] *= -1
            self._spins = self._spins.reshape(-1, self.n_copy).T
            self._spins = np.array([np.random.permutation(ss) for ss in self._spins])
        return self._spins

    @property
    def s_histo(self):
        return np.einsum('nj,ij->ni', self.s_prod, self.kappa) * self.prefactor

    @property
    def s_prod(self):
        return np.prod(self.spins.T[self.indices], axis=1).T

    @property
    def t_histo(self):
        return self.histogram * (1 - 2 * self.concentration)**2

    @property
    def sample_points(self):
        if self._sample_points is None:
            self._sample_points = self.kappa[self.histogram > self.min_sample_value]
        return self._sample_points

    def run_mc(self, n_steps=1000):
        current_value = np.inf
        results = []
        for iii in tqdm(range(n_steps)):
            index = np.random.choice(self.n_copy)
            spin_p = np.random.choice(np.where(self.spins[index] > 0)[0])
            spin_m = np.random.choice(np.where(self.spins[index] < 0)[0])
            self._spins[index, spin_p] *= -1
            self._spins[index, spin_m] *= -1
            s_diff = self.s_prod - (1 - 2 * self.concentration)**2
            phi = np.einsum('nj,ij->ni', s_diff, self.sample_points)
            new_value = np.sum(phi.sum(axis=0)**2 + np.sum(phi**2, axis=0))
            if new_value > current_value:
                self._spins[index, spin_p] *= -1
                self._spins[index, spin_m] *= -1
            else:
                current_value = new_value
                results.append([iii, current_value])
        return np.array(results)


class Input(GenericInput):
    concentration = InputField('concentration', 'Concentration', float)
        name,
        doc,
        data_type=None,
        fget=lambda x: x,
        fset=lambda x: x,
        default=None,


class SQS(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
