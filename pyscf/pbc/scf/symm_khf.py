#!/usr/bin/env python
#
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock for periodic systems with k-point sampling

See Also:
    hf.py : Hartree-Fock for periodic systems at a single k-point
'''

import sys
import time
from functools import reduce
import numpy as np
import scipy.linalg
import h5py
from pyscf.pbc.scf import hf as pbchf
from pyscf import lib
from pyscf.symm.point_group import get_stars
from pyscf.scf import hf
from pyscf.lib import logger
from pyscf.pbc.gto import ecp
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile
from pyscf.pbc import tools
from pyscf.pbc import df


def get_ovlp(mf, cell=None, kpts=None):
    '''Get the overlap AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    s = cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpts)
    cond = np.max(lib.cond(s))
    if cond * cell.precision > 1e2:
        prec = 1e2 / cond
        rmin = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
        if cell.rcut < rmin:
            logger.warn(cell, 'Singularity detected in overlap matrix.  '
                        'Integral accuracy may be not enough.\n      '
                        'You can adjust  cell.precision  or  cell.rcut  to '
                        'improve accuracy.  Recommended values are\n      '
                        'cell.precision = %.2g  or smaller.\n      '
                        'cell.rcut = %.4g  or larger.', prec, rmin)
    return lib.asarray(s)


def get_hcore(mf, cell=None, kpts=None):
    '''Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    return lib.asarray([pbchf.get_hcore(cell, k) for k in kpts])


def get_j(mf, cell, dm_kpts, kpts, kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.

    Kwargs:
        kpts_band : (k,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    return df.FFTDF(cell).get_jk(dm_kpts, kpts, kpts_band, with_k=False)[0]


def get_jk(mf, cell, dm_kpts, kpts, kpts_band=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point

    Kwargs:
        kpts_band : (3,) ndarray
            A list of arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    return df.FFTDF(cell).get_jk(dm_kpts, kpts, kpts_band, exxdiv=mf.exxdiv)

def get_fock(mf, h1e_kpts, s_kpts, vhf_kpts, dm_kpts, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    f_kpts = h1e_kpts + vhf_kpts
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts)
    if abs(level_shift_factor) > 1e-4:
        f_kpts = [hf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                  for k, s in enumerate(s_kpts)]
    return lib.asarray(f_kpts)

def get_fermi(mf, mo_energy_kpts=None, mo_occ_kpts=None):
    '''Fermi level
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    if mo_occ_kpts is None: mo_occ_kpts = mf.mo_occ
    nocc = np.count_nonzero(mo_occ_kpts != 0)
    fermi = np.sort(mo_energy_kpts.ravel())[nocc-1]
    return fermi

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    nkpts = len(mo_energy_kpts)
    nocc = (mf.cell.nelectron * nkpts) // 2

    # TODO: implement Fermi smearing and print mo_energy kpt by kpt
    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(np.double) * 2)

    if nocc < mo_energy.size:
        logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                    mo_energy[nocc-1], mo_energy[nocc])
        if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
            logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
    else:
        logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[k][mo_occ_kpts[k]> 0],
                         mo_energy_kpts[k][mo_occ_kpts[k]==0])
        np.set_printoptions(threshold=1000)

    return mo_occ_kpts


def get_grad(mo_coeff_kpts, mo_occ_kpts, fock):
    '''
    returns 1D array of gradients, like non K-pt version
    note that occ and virt indices of different k pts now occur
    in sequential patches of the 1D array
    '''
    nkpts = len(mo_occ_kpts)
    grad_kpts = [hf.get_grad(mo_coeff_kpts[k], mo_occ_kpts[k], fock[k])
                 for k in range(nkpts)]
    return np.hstack(grad_kpts)


def make_rdm1(mo_coeff_kpts, mo_occ_kpts):
    '''One particle density matrices for all k-points.

    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    '''
    nkpts = len(mo_occ_kpts)
    dm_kpts = [hf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k])
               for k in range(nkpts)]
    return lib.asarray(dm_kpts)


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    #for idm, dm in enumerate(dm_kpts):
    #    print idm, dm.flatten()[:2]
    nkpts = len(dm_kpts)
    #dm_kpts  = dm_kpts[mf.irr_kpts_idx]
    #h1e_kpts = h1e_kpts[mf.irr_kpts_idx]
    #vhf_kpts = vhf_kpts[mf.irr_kpts_idx]

    weighted_dm_kpts = 1./nkpts * dm_kpts
    #weighted_dm_kpts = np.einsum('w,wij->wij', mf.kpt_weights, dm_kpts)
    e1 = np.einsum('kij,kji', weighted_dm_kpts, h1e_kpts)
    e_coul = np.einsum('kij,kji', weighted_dm_kpts, vhf_kpts) * 0.5
    if abs(e_coul.imag > 1.e-7):
        raise RuntimeError("Coulomb energy has imaginary part, "
                           "something is wrong!", e_coul.imag)
    e1 = e1.real
    e_coul = e_coul.real
    logger.debug(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul


def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=True, **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    if with_meta_lowdin:
        return mf.mulliken_meta(mf.cell, dm, s=ovlp_ao, verbose=verbose)
    else:
        return mf.mulliken_pop(mf.cell, dm, s=ovlp_ao, verbose=verbose)


def mulliken_meta(cell, dm_ao, verbose=logger.DEBUG, pre_orth_method='ANO',
                  s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.

    Note this function only computes the Mulliken population for the gamma
    point density matrix.
    '''
    from pyscf.lo import orth
    if s is None:
        s = get_ovlp(cell)
    log = logger.new_logger(cell, verbose)
    log.note('Analyze output for the gamma point')
    log.note("KRHF mulliken_meta")
    dm_ao_gamma = dm_ao[0,:,:].real
    s_gamma = s[0,:,:].real
    c = orth.restore_ao_character(cell, pre_orth_method)
    orth_coeff = orth.orth_ao(cell, 'meta_lowdin', pre_orth_ao=c, s=s_gamma)
    c_inv = np.dot(orth_coeff.T, s_gamma)
    dm = reduce(np.dot, (c_inv, dm_ao_gamma, c_inv.T.conj()))

    log.note(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return hf.mulliken_pop(cell, dm, np.eye(orth_coeff.shape[0]), log)


def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_hcore() + mf.get_jk(mf.cell, dm)
    mo_coeff = []
    mo_energy = []
    for k, mo in enumerate(mo_coeff_kpts):
        mo1 = np.empty_like(mo)
        mo_e = np.empty_like(mo_occ_kpts[k])
        occidx = mo_occ_kpts[k] == 2
        viridx = ~occidx
        for idx in (occidx, viridx):
            if np.count_nonzero(idx) > 0:
                orb = mo[:,idx]
                f1 = reduce(np.dot, (orb.T.conj(), fock[k], orb))
                e, c = scipy.linalg.eigh(f1)
                mo1[:,idx] = np.dot(orb, c)
                mo_e[idx] = e
        mo_coeff.append(mo1)
        mo_energy.append(mo_e)
    return mo_energy, mo_coeff

def irr_transform(irr_array, op=-1):
    if op == 1.:
        return irr_array
    elif op == -1.:
        return irr_array.conj()
    else:
        raise NotImplementedError('No transform for operator ', op)

def init_guess_by_chkfile(cell, chkfile_name, project=True, kpts=None):
    '''Read the KHF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 3D ndarray
    '''
    from pyscf.pbc.scf import kuhf
    dm = kuhf.init_guess_by_chkfile(cell, chkfile_name, project, kpts)
    return dm[0] + dm[1]

def get_kpt_weights(kpt_stars):
    nelements_per_star = np.array([1. * len(star) for star, op in kpt_stars])
    return nelements_per_star / np.sum(nelements_per_star)

def get_irr_kpts(kpts, kpt_stars):
    kpts_idx, kpts = zip(*[(star[0], kpts[star[0]]) for star, op in kpt_stars])
    return list(kpts_idx), list(kpts)

class KSCF(hf.SCF):
    '''SCF class with k-point sampling.

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k-points,
    e.g. mo_coeff is (nkpts, nao, nao) ndarray

    Attributes:
        kpts : (nks,3) ndarray
            The sampling k-points in Cartesian coordinates, in units of 1/Bohr.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3)), exxdiv='ewald'):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        self.exxdiv = exxdiv
        self.kpts = kpts
        self.kpt_stars = get_stars(self.kpts, self.cell._atom, only_inversion=False)

        #kpt_star_weights = [(star[0], ikpt, len(star)) for star, op in self.kpt_stars for ikpt in star]
        #print kpt_star_weights
        #self.kpt_star_weights = sorted(kpt_star_weights, key=lambda x: x[1])
        #self.kpt_weights = np.array([1. * x[2] for x in kpt_star_weights])
        #self.kpt_weights = np.array([1. * len(star[0]) for star in self.kpt_stars for ikpt in star[0]])
        self.kpt_weights = get_kpt_weights(self.kpt_stars) #np.array([len(star) for star, op in self.kpt_stars])
        self.with_df.kpt_weights = self.kpt_weights
        print self.kpt_weights
        self.irr_kpts_idx, self.irr_kpts = get_irr_kpts(self.kpts, self.kpt_stars)
        print self.irr_kpts_idx
        print self.kpt_stars
        #print self.kpt_star_weights
        print self.irr_kpts
        self.with_df.irr_kpts = self.irr_kpts

        self.direct_scf = False

        self.exx_built = False
        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv', 'with_df'])

    @property
    def kpts(self):
        return self.with_df.kpts
    @kpts.setter
    def kpts(self, x):
        self.with_df.kpts = np.reshape(x, (-1,3))

    @property
    def mo_energy_kpts(self):
        return self.mo_energy

    @property
    def mo_coeff_kpts(self):
        return self.mo_coeff

    @property
    def mo_occ_kpts(self):
        return self.mo_occ

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'N kpts = %d', len(self.kpts))
        logger.debug(self, 'kpts = %s', self.kpts)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        #if self.exxdiv == 'vcut_ws':
        #    if self.exx_built is False:
        #        self.precompute_exx()
        #    logger.info(self, 'WS alpha = %s', self.exx_alpha)
        if isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald':
            madelung = tools.pbc.madelung(self.cell, [self.kpts])
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung/cell.vol = %.12g',
                        madelung*self.cell.nelectron * -.5)
        logger.info(self, 'DF object = %s', self.with_df)
        self.with_df.dump_flags()
        return self

    def check_sanity(self):
        hf.SCF.check_sanity(self)
        self.with_df.check_sanity()
        if (isinstance(self.exxdiv, str) and self.exxdiv.lower() != 'ewald' and
            isinstance(self.with_df, df.df.DF)):
            logger.warn(self, 'exxdiv %s is not supported in DF or MDF',
                        self.exxdiv)
        return self

    def build(self, cell=None):
        hf.SCF.build(self, cell)
        #if self.exxdiv == 'vcut_ws':
        #    self.precompute_exx()

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None:
            cell = self.cell
        dm_kpts = None
        if key.lower() == '1e':
            dm_kpts = self.init_guess_by_1e(cell)
        elif getattr(cell, 'natm', 0) == 0:
            logger.info(self, 'No atom found in cell. Use 1e initial guess')
            dm_kpts = self.init_guess_by_1e(cell)
        elif key.lower() == 'atom':
            dm = self.init_guess_by_atom(cell)
        elif key.lower().startswith('chk'):
            try:
                dm_kpts = self.from_chk()
            except (IOError, KeyError):
                logger.warn(self, 'Fail in reading %s. Use MINAO initial guess',
                            self.chkfile)
                dm = self.init_guess_by_minao(cell)
        else:
            dm = self.init_guess_by_minao(cell)

        if dm_kpts is None:
            dm_kpts = lib.asarray([dm]*len(self.kpts))

        if cell.dimension < 3:
            ne = np.einsum('kij,kji->k', dm_kpts, self.get_ovlp(cell)).real
            if np.any(abs(ne - cell.nelectron) > 1e-7):
                logger.warn(self, 'Big error detected in the electron number '
                            'of initial guess density matrix (Ne/cell = %g)!\n'
                            '  This can cause huge error in Fock matrix and '
                            'lead to instability in SCF for low-dimensional '
                            'systems.\n  DM is normalized to correct number '
                            'of electrons', ne.mean())
                dm_kpts *= cell.nelectron / ne.reshape(-1,1,1)
        return dm_kpts

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        return hf.SCF.init_guess_by_1e(self, cell)

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if cell.pseudo:
            nuc = lib.asarray(self.with_df.get_pp(kpts))
        else:
            nuc = lib.asarray(self.with_df.get_nuc(kpts))
        if len(cell._ecpbas) > 0:
            nuc += lib.asarray(ecp.ecp_int(cell, kpts))
        t = lib.asarray(cell.pbc_intor('int1e_kin_sph', 1, 1, kpts))
        hcore = nuc + t

        #hcore = []
        ## Unfold irreducible k-points into the full k-point array
        #for irr_kpt, kpt, _ in self.kpt_star_weights:
        #    if irr_kpt == kpt:
        #        hcore.append(irr_transform(irr_hcore[irr_kpt,:,:], op=1))
        #    else:
        #        hcore.append(irr_transform(irr_hcore[irr_kpt,:,:], op=-1))
# self.kpt_star_weights
#[[0.  0.  0.5]
# [1.  1.  0.5]
# [2.  2.  0.5]
# [3.  3.  0.5]
# [3.  4.  0.5]
# [2.  5.  0.5]
# [1.  6.  0.5]
# [0.  7.  0.5]]
#[0, 1, 2, 3]

        hcore = lib.asarray(hcore)
        return hcore

    get_ovlp = get_ovlp
    get_fock = get_fock
    get_occ = get_occ
    energy_elec = energy_elec
    get_fermi = get_fermi

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band, with_k=False)[0]
        logger.timer(self, 'vj', *cpu0)
        return vj

    def get_k(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)[1]

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj, vk = self.with_df.get_jk(dm_kpts, hermi, self.kpts, kpts_band,
                                     exxdiv=self.exxdiv, with_k=False)
        vk = 0.0 * vj
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        vj, vk = self.get_jk(cell, dm_kpts, hermi, self.kpts, kpts_band)
        return vj - vk * .5

    def analyze(self, verbose=None, with_meta_lowdin=True, **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, with_meta_lowdin, **kwargs)

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)
        return get_grad(mo_coeff_kpts, mo_occ_kpts, fock)

    def eig(self, h_kpts, s_kpts):
        nkpts = len(h_kpts)
        eig_kpts = []
        mo_coeff_kpts = []

        for k in range(nkpts):
            e, c = self._eigh(h_kpts[k], s_kpts[k])
            eig_kpts.append(e)
            mo_coeff_kpts.append(c)
        return eig_kpts, mo_coeff_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        if mo_coeff_kpts is None:
            # Note: this is actually "self.mo_coeff_kpts"
            # which is stored in self.mo_coeff of the scf.hf.RHF superclass
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            # Note: this is actually "self.mo_occ_kpts"
            # which is stored in self.mo_occ of the scf.hf.RHF superclass
            mo_occ_kpts = self.mo_occ

        #if mo_coeff_kpts is not None:
        #    print 'make_rdm1'
        #    print mo_coeff_kpts
        #    print mo_occ_kpts
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts)

    def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at the given (arbitrary) 'band' k-points.

        Returns:
            mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
                Band orbitals psi_n(k)
        '''
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = (kpts_band.ndim == 1)
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_hcore(cell, kpts_band)
        fock = fock + self.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)
        mo_energy, mo_coeff = self.eig(fock, s1e)
        if single_kpt_band:
            mo_energy = mo_energy[0]
            mo_coeff = mo_coeff[0]
        return mo_energy, mo_coeff

    def init_guess_by_chkfile(self, chk=None, project=True, kpts=None):
        if chk is None: chk = self.chkfile
        if kpts is None: kpts = self.kpts
        return init_guess_by_chkfile(self.cell, chk, project, kpts)
    def from_chk(self, chk=None, project=True, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def dump_chk(self, envs):
        hf.SCF.dump_chk(self, envs)
        if self.chkfile:
            with h5py.File(self.chkfile) as fh5:
                fh5['scf/kpts'] = self.kpts
        return self

    def mulliken_meta(self, cell=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method='ANO', s=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(cell)
        return mulliken_meta(cell, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    canonicalize = canonicalize

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import df_jk
        return df_jk.density_fit(self, auxbasis, with_df)

    def mix_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import mdf_jk
        return mdf_jk.density_fit(self, auxbasis, with_df)

    def stability(self, internal=True, external=False, verbose=None):
        from pyscf.pbc.scf.stability import rhf_stability
        return rhf_stability(self, internal, external, verbose)

    def newton(self):
        from pyscf.pbc.scf import newton_ah
        return newton_ah.newton(self)

    def x2c1e(self):
        from pyscf.pbc.scf import x2c
        return x2c.sfx2c1e(self)

KRHF = KSCF


if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = '321g'
    cell.a = np.eye(3) * 3
    cell.gs = [25] * 3
    tol = 1e-14
    cell.conv_tol = tol
    cell.precision = tol
    cell.verbose = 5
    cell.build()
    kpt = [3,3,3]
    mf = KRHF(cell, cell.make_kpts(kpt, with_gamma_point=False))
    mf.conv_tol_grad = tol*1000
    e = mf.kernel()
    if kpt == [2,2,2]:
        #assert(e - -5.86521437670938 < 1e-12)
        assert(abs(e - -3.33158932225045) < 1e-12)  # with_k = False
    elif kpt == [2,2,3]:
        #assert(abs(e - -5.8652134686146) < 1e-12)
        assert(abs(e - -3.33158895465433) < 1e-12)
    elif kpt == [3,3,3]:
        #assert(abs(e - -5.86501194631279) < 1e-12)
        assert(abs(e - -3.33012203400506) < 1e-12)
    mf.analyze()

#   0 (-0.250 -0.250 -0.333)   [-1.29905467 -0.78523682] [1.27053927 2.03092942]
#   1 (-0.250 -0.250  0.000)   [-1.30005369 -0.78679906] [1.24686667 2.00437288]
#   2 (-0.250 -0.250  0.333)   [-1.29905468 -0.78523682] [1.27053929 2.03092939]
#   3 (-0.250  0.250 -0.333)   [-1.29905467 -0.78523682] [1.27053927 2.03092942]
#   4 (-0.250  0.250  0.000)   [-1.30005369 -0.78679906] [1.24686667 2.00437288]
#   5 (-0.250  0.250  0.333)   [-1.29905468 -0.78523682] [1.27053929 2.03092939]
#   6 ( 0.250 -0.250 -0.333)   [-1.29905468 -0.78523682] [1.27053929 2.03092939]
#   7 ( 0.250 -0.250  0.000)   [-1.30005369 -0.78679906] [1.24686667 2.00437288]
#   8 ( 0.250 -0.250  0.333)   [-1.29905467 -0.78523682] [1.27053927 2.03092942]
#   9 ( 0.250  0.250 -0.333)   [-1.29905468 -0.78523682] [1.27053929 2.03092939]
#  10 ( 0.250  0.250  0.000)   [-1.30005369 -0.78679906] [1.24686667 2.00437288]
#  11 ( 0.250  0.250  0.333)   [-1.29905467 -0.78523682] [1.27053927 2.03092942]

#   0 (-0.333 -0.333 -0.333)   [-1.29770498 -0.79046232] [1.23704826 2.00046713]
#   1 (-0.333 -0.333  0.000)   [-1.29865709 -0.79213149] [1.21405719 1.97375878]
#   2 (-0.333 -0.333  0.333)   [-1.29770499 -0.79046232] [1.23704828 2.00046712]
#   3 (-0.333  0.000 -0.333)   [-1.29865634 -0.79213085] [1.21405389 1.97375618]
#   4 (-0.333  0.000  0.000)   [-1.29963524 -0.79383438] [1.19123857 1.94736928]
#   5 (-0.333  0.000  0.333)   [-1.29865635 -0.79213085] [1.21405391 1.97375616]
#   6 (-0.333  0.333 -0.333)   [-1.29770498 -0.79046232] [1.23704826 2.00046713]
#   7 (-0.333  0.333  0.000)   [-1.29865709 -0.79213149] [1.21405719 1.97375878]
#   8 (-0.333  0.333  0.333)   [-1.29770499 -0.79046232] [1.23704828 2.00046712]
#   9 ( 0.000 -0.333 -0.333)   [-1.30066431 -0.77303541] [1.36015866 2.1504475 ]
#  10 ( 0.000 -0.333  0.000)   [-1.30173003 -0.77434799] [1.3348854  2.12360907]
#  11 ( 0.000 -0.333  0.333)   [-1.30066431 -0.77303541] [1.36015866 2.1504475 ]
#  12 ( 0.000  0.000 -0.333)   [-1.30172929 -0.77434733] [1.33488193 2.12360629]
#  13 ( 0.000  0.000  0.000)   [-1.30282345 -0.77568886] [1.30985268 2.09705455]
#  14 ( 0.000  0.000  0.333)   [-1.30172929 -0.77434733] [1.33488193 2.12360629]
#  15 ( 0.000  0.333 -0.333)   [-1.30066431 -0.77303541] [1.36015866 2.1504475 ]
#  16 ( 0.000  0.333  0.000)   [-1.30173003 -0.77434799] [1.3348854  2.12360907]
#  17 ( 0.000  0.333  0.333)   [-1.30066431 -0.77303541] [1.36015866 2.1504475 ]
#  18 ( 0.333 -0.333 -0.333)   [-1.29770499 -0.79046232] [1.23704828 2.00046712]
#  19 ( 0.333 -0.333  0.000)   [-1.29865709 -0.79213149] [1.21405719 1.97375878]
#  20 ( 0.333 -0.333  0.333)   [-1.29770498 -0.79046232] [1.23704826 2.00046713]
#  21 ( 0.333  0.000 -0.333)   [-1.29865635 -0.79213085] [1.21405391 1.97375616]
#  22 ( 0.333  0.000  0.000)   [-1.29963524 -0.79383438] [1.19123857 1.94736928]
#  23 ( 0.333  0.000  0.333)   [-1.29865634 -0.79213085] [1.21405389 1.97375618]
#  24 ( 0.333  0.333 -0.333)   [-1.29770499 -0.79046232] [1.23704828 2.00046712]
#  25 ( 0.333  0.333  0.000)   [-1.29865709 -0.79213149] [1.21405719 1.97375878]
#  26 ( 0.333  0.333  0.333)   [-1.29770498 -0.79046232] [1.23704826 2.00046713]

#converged SCF energy = -3.33158895465433
#Analyze output for the gamma point
#KRHF mulliken_meta
# ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **
# ** Mulliken pop  **
#pop of  0He 1s        1.97094
#pop of  0He 2s        0.03111
#pop of  1He 1s        1.93819
#pop of  1He 2s        0.06578
# ** Mulliken atomic charges  **
#charge of  0He =     -0.00205
#charge of  1He =     -0.00397
#[array([[ 0.23126742-0.0029047 j, -0.26326892-0.00682169j,
#         1.16379269-0.00292807j,  0.34920021-0.00338418j],
#       [ 0.42519711+0.02420935j, -0.75799201+0.04684957j,
#        -1.10019212-0.0450628 j, -0.2740312 +0.27599331j],
#       [ 0.18550681-0.01756726j,  0.19863157-0.0490722 j,
#        -0.11467055-0.29053642j,  0.50813597+1.08022103j],
#       [ 0.4730487 -0.07415497j,  0.78681668-0.13121987j,
#         0.37664597+0.18258541j, -0.3458082 -0.980555  j]]), array([[ 0.22772779-0.00298994j, -0.25987736-0.00696958j,
#         1.16450532-0.00315461j,  0.34316984-0.0025693 j],
#       [ 0.42568706+0.02474679j, -0.76062599+0.04770706j,
#        -1.09024181-0.0430699 j, -0.27041649+0.27333104j],
#       [ 0.18294155-0.01749683j,  0.1951743 -0.04913958j,
#        -0.10921309-0.28695139j,  0.4928214 +1.08714555j],
#       [ 0.47413988-0.0754532 j,  0.78751487-0.13317444j,
#         0.36842724+0.17727311j, -0.33075326-0.97774632j]]), array([[ 0.23126742-0.00290452j, -0.26326893-0.00682192j,
#         1.16379269-0.00292844j,  0.3492002 -0.00338431j],
#       [ 0.42519712+0.02420911j, -0.75799198+0.04685011j,
#        -1.10019217-0.04506207j, -0.27403097+0.27599346j],
#       [ 0.18550682-0.01756713j,  0.19863159-0.04907203j,
#        -0.11467066-0.29053637j,  0.50813644+1.08022082j],
#       [ 0.47304867-0.07415517j,  0.78681659-0.13122037j,
#         0.3766461 +0.1825851 j, -0.34580887-0.98055477j]]), array([[ 0.23126742-0.0029047 j, -0.26326892-0.00682169j,
#         1.16379269-0.00292807j,  0.34920021-0.00338418j],
#       [ 0.42519711+0.02420935j, -0.75799201+0.04684957j,
#        -1.10019212-0.0450628 j, -0.2740312 +0.27599331j],
#       [ 0.18550681-0.01756726j,  0.19863157-0.0490722 j,
#        -0.11467055-0.29053642j,  0.50813597+1.08022103j],
#       [ 0.4730487 -0.07415497j,  0.78681668-0.13121987j,
#         0.37664597+0.18258541j, -0.3458082 -0.980555  j]]), array([[ 0.22772779-0.00298994j, -0.25987736-0.00696958j,
#         1.16450532-0.00315461j,  0.34316984-0.0025693 j],
#       [ 0.42568706+0.02474679j, -0.76062599+0.04770706j,
#        -1.09024181-0.0430699 j, -0.27041649+0.27333104j],
#       [ 0.18294155-0.01749683j,  0.1951743 -0.04913958j,
#        -0.10921309-0.28695139j,  0.4928214 +1.08714555j],
#       [ 0.47413988-0.0754532 j,  0.78751487-0.13317444j,
#         0.36842724+0.17727311j, -0.33075326-0.97774632j]]), array([[ 0.23126742-0.00290452j, -0.26326893-0.00682192j,
#         1.16379269-0.00292844j,  0.3492002 -0.00338431j],
#       [ 0.42519712+0.02420911j, -0.75799198+0.04685011j,
#        -1.10019217-0.04506207j, -0.27403097+0.27599346j],
#       [ 0.18550682-0.01756713j,  0.19863159-0.04907203j,
#        -0.11467066-0.29053637j,  0.50813644+1.08022082j],
#       [ 0.47304867-0.07415517j,  0.78681659-0.13122037j,
#         0.3766461 +0.1825851 j, -0.34580887-0.98055477j]]), array([[ 0.23126742+0.00290452j, -0.26326893+0.00682192j,
#         1.16379269+0.00292844j,  0.3492002 +0.00338431j],
#       [ 0.42519712-0.02420911j, -0.75799198-0.04685011j,
#        -1.10019217+0.04506207j, -0.27403097-0.27599346j],
#       [ 0.18550682+0.01756713j,  0.19863159+0.04907203j,
#        -0.11467066+0.29053637j,  0.50813644-1.08022082j],
#       [ 0.47304867+0.07415517j,  0.78681659+0.13122037j,
#         0.3766461 -0.1825851 j, -0.34580887+0.98055477j]]), array([[ 0.22772779+0.00298994j, -0.25987736+0.00696958j,
#         1.16450532+0.00315461j,  0.34316984+0.0025693 j],
#       [ 0.42568706-0.02474679j, -0.76062599-0.04770706j,
#        -1.09024181+0.0430699 j, -0.27041649-0.27333104j],
#       [ 0.18294155+0.01749683j,  0.1951743 +0.04913958j,
#        -0.10921309+0.28695139j,  0.4928214 -1.08714555j],
#       [ 0.47413988+0.0754532 j,  0.78751487+0.13317444j,
#         0.36842724-0.17727311j, -0.33075326+0.97774632j]]), array([[ 0.23126742+0.0029047 j, -0.26326892+0.00682169j,
#         1.16379269+0.00292807j,  0.34920021+0.00338418j],
#       [ 0.42519711-0.02420935j, -0.75799201-0.04684957j,
#        -1.10019212+0.0450628 j, -0.2740312 -0.27599331j],
#       [ 0.18550681+0.01756726j,  0.19863157+0.0490722 j,
#        -0.11467055+0.29053642j,  0.50813597-1.08022103j],
#       [ 0.4730487 +0.07415497j,  0.78681668+0.13121987j,
#         0.37664597-0.18258541j, -0.3458082 +0.980555  j]]), array([[ 0.23126742+0.00290452j, -0.26326893+0.00682192j,
#         1.16379269+0.00292844j,  0.3492002 +0.00338431j],
#       [ 0.42519712-0.02420911j, -0.75799198-0.04685011j,
#        -1.10019217+0.04506207j, -0.27403097-0.27599346j],
#       [ 0.18550682+0.01756713j,  0.19863159+0.04907203j,
#        -0.11467066+0.29053637j,  0.50813644-1.08022082j],
#       [ 0.47304867+0.07415517j,  0.78681659+0.13122037j,
#         0.3766461 -0.1825851 j, -0.34580887+0.98055477j]]), array([[ 0.22772779+0.00298994j, -0.25987736+0.00696958j,
#         1.16450532+0.00315461j,  0.34316984+0.0025693 j],
#       [ 0.42568706-0.02474679j, -0.76062599-0.04770706j,
#        -1.09024181+0.0430699 j, -0.27041649-0.27333104j],
#       [ 0.18294155+0.01749683j,  0.1951743 +0.04913958j,
#        -0.10921309+0.28695139j,  0.4928214 -1.08714555j],
#       [ 0.47413988+0.0754532 j,  0.78751487+0.13317444j,
#         0.36842724-0.17727311j, -0.33075326+0.97774632j]]), array([[ 0.23126742+0.0029047 j, -0.26326892+0.00682169j,
#         1.16379269+0.00292807j,  0.34920021+0.00338418j],
#       [ 0.42519711-0.02420935j, -0.75799201-0.04684957j,
#        -1.10019212+0.0450628 j, -0.2740312 -0.27599331j],
#       [ 0.18550681+0.01756726j,  0.19863157+0.0490722 j,
#        -0.11467055+0.29053642j,  0.50813597-1.08022103j],
#       [ 0.4730487 +0.07415497j,  0.78681668+0.13121987j,
#         0.37664597-0.18258541j, -0.3458082 +0.980555  j]])]
