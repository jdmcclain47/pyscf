#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
#
# The symmetry detection method implemented here is not strictly follow the
# point group detection flowchart.  The detection is based on the degeneracy
# of cartesian basis of multipole momentum, eg
# http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=604&option=4
# see the column of "linear functions, quadratic functions and cubic functions".
#
# Different point groups have different combinations of degeneracy for each
# type of cartesian functions.  Based on the degeneracy of cartesian function
# basis, one can quickly filter out a few candidates of point groups for the
# given molecule.  Regular operations (rotation, mirror etc) can be applied
# next to identify the symmetry.  Current implementation only checks the
# rotation functions and it's roughly enough for D2h and subgroups.
#
# There are special cases this detection method may break down, eg two H8 cube
# molecules sitting on the same center but with random orientation.  The
# system is in C1 while this detection method gives O group because the
# 3 rotation bases are degenerated.  In this case, the code use the regular
# method (point group detection flowchart) to detect the point group.
#

import sys
import re
import numpy
import scipy.linalg
from pyscf.gto import mole
from pyscf.lib import norm
from pyscf.lib import logger
import pyscf.symm.param

TOLERANCE = 1e-5

def parallel_vectors(v1, v2, tol=TOLERANCE):
    """

    Finds whether two lines are parallel.

    Parameters
    ----------
    v1 : array_like
        First vector.
    v2 : array_like
        Second vector.
    tol : scalar
        Tolerance for whether two lines are considered parallel.

    Returns
    -------
    is_parallel : bool
        Whether the two vectors are parallel.

    Notes
    -----
    Note that a vector of length zero is defined to be parallel to any other
    vector.

    """
    if numpy.allclose(v1, 0, atol=tol) or numpy.allclose(v2, 0, atol=tol):
        return True
    else:
        cos = numpy.dot(_normalize(v1), _normalize(v2))
        return (abs(cos-1) < TOLERANCE) | (abs(cos+1) < TOLERANCE)

def argsort_coords(coords, decimals=None):
    """

    Returns the indices that would sort a set of coordinates, where the
    coordinates are first sorted by x-, then y-, and finally z-coordinate.

    Parameters
    ----------
    coords : array_like
        `(N,3)` array of coordinates.
    decimals : scalar
        Precision for rounding in sort.

    Returns
    -------
    idx : `(N,)` ndarray of ints
        Index to sort `coords`.

    """
    if coords.shape[1] != 3:
        raise ValueError("Expected second dimension of coords to be 3")

    if decimals is None:
        decimals = int(-numpy.log10(TOLERANCE)) - 1
    coords = numpy.around(coords, decimals=decimals)
    idx = numpy.lexsort((coords[:,2], coords[:,1], coords[:,0]))
    return idx

def sort_coords(coords, decimals=None):
    """

    Returns a sorted set of coordinates, where the coordinates are first sorted
    by x-, then y-, and finally z-coordinate.

    Parameters
    ----------
    coords : array_like
        `(N,3)` array of coordinates.
    decimals : scalar
        Precision for rounding in sort.

    Returns
    -------
    sorted_coords : ndarray
        Sorted `coords` array.

    """
    if decimals is None:
        decimals = int(-numpy.log10(TOLERANCE)) - 1
    coords = numpy.asarray(coords)
    idx = argsort_coords(coords, decimals=decimals)
    return coords[idx]

def rotation_mat(vec, theta):
    """

    Gives a `(3,3)` rotation matrix for the rotation of `theta` radians about
    an axis `vec`.  The angle is in the direction as dictated by the Right-hand
    rule (see ref. https://en.wikipedia.org/wiki/Right-hand_rule).

    Parameters
    ----------
    vec : array_like
        `(3,)` vector array about which to rotate.
    theta : scalar
        Angle to rotate in radians.

    Returns
    -------
    rotation_mat : `(3,3)` ndarray
        3D rotation matrix.

    Notes
    -----
    See ref. `http://en.wikipedia.org/wiki/Rotation_matrix` for more details.

    """
    vec = _normalize(vec)
    uu = vec.reshape(-1,1) * vec.reshape(1,-1)
    ux = numpy.array((
        ( 0     ,-vec[2], vec[1]),
        ( vec[2], 0     ,-vec[0]),
        (-vec[1], vec[0], 0     )))
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    r = c * numpy.eye(3) + s * ux + (1-c) * uu
    return r

def householder(vec):
    """

    Gives the `(3,3)` Householder matrix, `P`, for reflection across a given
    hyperplane described by a normal vector `vec`.  The matrix `P` is given by

    .. math:: P = I - 2vv^T

    Parameters
    ----------
    vec : array_like
        `(3,)` vector array about which to rotate.

    Returns
    -------
    household_mat : `(3,3)` ndarray
        3D Householder matrix.

    Notes
    -----
    See ref. `http://en.wikipedia.org/wiki/Householder_transformation`
    for more details.

    """
    vec = _normalize(vec)
    return numpy.eye(3) - vec[:,None]*vec*2

def closest_axes_index(axes, ref_axes):
    """

    This function finds a mapping of axes from a coordinate system `axes`
    to axes from a reference coordinate system `ref_axes` such that there is
    maximal overlap of axes with the reference coordinate system (not
    necessarily the standard Cartesian coordinate system).

    Parameters
    ----------
    axes : array_like
        `(3,3)` matrix describing a coordinate system where each row is an
        axis in the coordinate system.
    ref_axes : array_like
        `(3,3)` matrix describing a reference coordinate system where each row
        is an axis in the coordinate system.

    Returns
    -------
    axes_idx : `(3,)` list
        mapping from old coordinate system axes to new coordinate system axes.

    Notes
    -----
    This is a greedy search.  Not guaranteed to give a true maximum overlap.

    """
    # Here we use x,y,z as aliases for the 1st,2nd,3rd axis
    xcomp, ycomp, zcomp = numpy.einsum('ix,jx->ji', axes, ref_axes)
    z_id = numpy.argmax(abs(zcomp))

    # Remove z-component and maximize over x.
    xcomp[z_id] = ycomp[z_id] = 0
    x_id = numpy.argmax(abs(xcomp))

    # Remove x-component and maximize over y.
    ycomp[x_id] = 0
    y_id = numpy.argmax(abs(ycomp))
    return x_id, y_id, z_id

def align_axes(axes, ref_axes):
    """

    This function returns a reordering of the coordinate system `axes`
    such that each component vector has maximal overlap with a reference
    coordinate system `ref_axes`.

    Parameters
    ----------
    axes : ndarray
        `(3,3)` matrix describing a coordinate system where each row is an
        axis in the coordinate system.
    ref_axes : array_like
        `(3,3)` matrix describing a reference coordinate system where each
        row is an axis in the coordinate system.

    Returns
    -------
    new_axes : ndarray
        `(3,3)` matrix describing the reordered coordinate system.

    Notes
    -----
    This is a greedy search.  Not guaranteed to give a true maximum overlap.

    """
    x_id, y_id, z_id = closest_axes_index(axes, ref_axes)
    new_axes = axes[[x_id,y_id,z_id]]
    # Make the resulting coordinate system obey the 'Right-hand Rule'
    if numpy.linalg.det(new_axes) < 0:
        new_axes = axes[[y_id,x_id,z_id]]
    return new_axes


def detect_symm(atoms, basis=None, verbose=logger.WARN):
    """

    Detect the point group symmetry for a given molecule.

    Return group name, charge center, and nex_axis (three rows for x,y,z)

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    """
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    tol = TOLERANCE / numpy.sqrt(1+len(atoms))
    decimals = int(-numpy.log10(tol))
    log.debug('geometry tol = %g', tol)

    # Set up symmetry object to handle all the possible symmetry operations
    # on the input atoms and optional basis set.
    rawsys = SymmSys(atoms, basis)

    # Obtain the eigenvalues of the dipole moment of the atomic-charge
    # distribution.
    w1, u1 = rawsys.cartesian_tensor(1)
    axes = u1.T
    log.debug('principal inertia moments %s', w1)

    # If we rotate the group in any direction and get itself (the SO(3) group),
    # then this is the same as having all zero eigenvalues.
    if numpy.allclose(w1, 0, atol=tol):
        gpname = 'SO3'
        return gpname, rawsys.charge_center, numpy.eye(3)

    # Linear molecule: If two directions are zero, then we only have one
    # principal axis.
    elif numpy.allclose(w1[:2], 0, atol=tol):

        # Check for inversion symmetry
        if rawsys.has_icenter():
            gpname = 'Dooh'
        else:
            gpname = 'Coov'
        return gpname, rawsys.charge_center, axes

    else:
        w1_degeneracy = _degeneracy(w1, decimals) # Get eigenvalue degeneracy

        n = None
        c2x = None
        mirrorx = None
        # Check for a triply-degenerate dipole element present in the
        # tetrahedral (T), octahedral (O), and icosohedral (I) groups.
        # More information found on http://symmetry.jacobs-university.de/pubs/pub4.html
        if 3 in w1_degeneracy: # T, O, I
            # Because rotation vectors Rx Ry Rz are 3-degenerated representation.
            # See http://www.webqc.org/symmetrypointgroup-td.html
            w2, u2 = rawsys.cartesian_tensor(2)
            w3, u3 = rawsys.cartesian_tensor(3)
            w2_degeneracy = _degeneracy(w2, decimals)
            w3_degeneracy = _degeneracy(w3, decimals)

            log.debug('2d tensor %s', w2)
            log.debug('3d tensor %s', w3)

            # Icosohedral group:
            #     Dipole    : T group (degeneracy 3)
            #     Quadrupole: H group (degeneracy 5)
            #     Octopole  : G group (degeneracy 4)
            if (5 in w2_degeneracy and
                4 in w3_degeneracy and len(w3_degeneracy) == 3):  # I group
                gpname, new_axes = _search_i_group(rawsys)
                if gpname is not None:
                    return gpname, rawsys.charge_center, _refine(new_axes)

            # Tetrahedral & Octahedral group:
            #     Quadrupole: T group (degeneracy 3)
            elif 3 in w2_degeneracy and len(w2_degeneracy) <= 3:  # T/O group
                gpname, new_axes = _search_ot_group(rawsys)
                if gpname is not None:
                    return gpname, rawsys.charge_center, _refine(new_axes)

        # All except D2h, C2h, C2v, D2, C2, Cs, Ci, or C1 will have at
        # least two degenerate rotation axes (as shown from jacobs.de ref).
        elif 2 in w1_degeneracy:
            # Check if we already have our x, y, or z principal axis
            # with the dipole eigenvectors.
            if numpy.allclose(w1[1], w1[2], atol=tol):
                axes = axes[[1,2,0]]
            n = rawsys.search_highest_cn(zaxis=axes[2])[1]
            if n == 1:  # we will need to exhaustively search for a
                        # rotation axis
                n = None
            else:
                c2x = rawsys.search_c2x(axes[2], n)
                mirrorx = rawsys.search_mirrorx(n, axes[2])

        else:  # Label it as D2h or a subgroup
            n = -1

        # This is not in the I/O/T group, and can contain at most one
        # 3-fold or higher rotation axis.  Exhaustively search for all
        # possible rotation axes and put this as the final z-axis.
        #
        # Remaining possible symmetries:
        #    C1, Cs, Ci, Cnv, Cnh, Cn, Dnh, Dnd, Sn
        if n is None:
            zaxis, n = rawsys.search_highest_cn()
            if n > 1:  # Cnv, Cnh, Cn, Dnh, Dnd, Sn
                c2x = rawsys.search_c2x(zaxis, n)
                mirrorx = rawsys.search_mirrorx(n, zaxis)

                # Make other axis components based on whether C2
                # or mirror axes were found
                if c2x is not None:
                    axes = _make_axes_from_2d(zaxis, c2x)
                elif mirrorx is not None:
                    axes = _make_axes_from_2d(zaxis, mirrorx)
                else:
                    for axis in numpy.eye(3): # Build standard cartesian
                        if not parallel_vectors(axis, zaxis):
                            axes = _make_axes_from_2d(zaxis, axis)
                            break
            else:  # Ci or Cs or C1 with degenerated w1
                mirror = rawsys.search_mirrorx(1, None)  # Exhaustive search
                if mirror is not None:
                    xaxis = numpy.array((1.,0.,0.))
                    axes = _make_axes_from_2d(mirror, xaxis)
                else:
                    axes = numpy.eye(3)

        log.debug('Highest C_n = C%d', n)
        if n >= 2:
            if c2x is not None:
                if rawsys.has_mirror(axes[2]):
                    gpname = 'D%dh' % n
                elif rawsys.has_icenter():
                    gpname = 'D%dd' % n
                else:
                    gpname = 'D%d' % n
                yaxis = numpy.cross(axes[2], c2x)
                axes = _make_axes_from_2d(axes[2], c2x)
            elif mirrorx is not None:
                gpname = 'C%dv' % n
                axes = _make_axes_from_2d(axes[2], mirrorx)
            elif rawsys.has_mirror(axes[2]):
                gpname = 'C%dh' % n
            elif all(rawsys.invariant_to_op(numpy.dot(rotation_mat(axes[2], numpy.pi/n),
                                                    householder(axes[2])))): # improper rotation
                gpname = 'S%d' % (n*2)
            else:
                gpname = 'C%d' % n
            return gpname, rawsys.charge_center, _refine(axes)

        else: # n = -1 (D2h and subgroups) or 1
            is_c2x = rawsys.has_rotation(axes[0], 2)
            is_c2y = rawsys.has_rotation(axes[1], 2)
            is_c2z = rawsys.has_rotation(axes[2], 2)
            if is_c2z and is_c2x and is_c2y:
                if rawsys.has_icenter():
                    gpname = 'D2h'
                else:
                    gpname = 'D2'
                # Align axes close to standard cartesian axis
                axes = align_axes(axes, numpy.eye(3))
            elif is_c2z or is_c2x or is_c2y:
                if is_c2x:
                    axes = axes[[1,2,0]]
                if is_c2y:
                    axes = axes[[2,0,1]]
                if rawsys.has_mirror(axes[2]):
                    gpname = 'C2h'
                elif rawsys.has_mirror(axes[0]):
                    gpname = 'C2v'
                else:
                    gpname = 'C2'
            else:
                if rawsys.has_icenter():
                    gpname = 'Ci'
                elif rawsys.has_mirror(axes[0]):
                    gpname = 'Cs'
                    axes = axes[[1,2,0]]
                elif rawsys.has_mirror(axes[1]):
                    gpname = 'Cs'
                    axes = axes[[2,0,1]]
                elif rawsys.has_mirror(axes[2]):
                    gpname = 'Cs'
                else:
                    gpname = 'C1'
    return gpname, rawsys.charge_center, axes


# reduce to D2h and its subgroups
# FIXME, CPL, 209, 506
def subgroup(gpname, axes):
    if gpname in ('D2h', 'D2' , 'C2h', 'C2v', 'C2' , 'Ci' , 'Cs' , 'C1'):
        return gpname, axes
    elif gpname in ('SO3',):
        #return 'D2h', align_axes(axes, numpy.eye(3))
        return 'Dooh', axes
    elif gpname in ('Dooh',):
        #return 'D2h', align_axes(axes, numpy.eye(3))
        return 'Dooh', axes
    elif gpname in ('Coov',):
        #return 'C2v', axes
        return 'Coov', axes
    elif gpname in ('Oh',):
        return 'D2h', align_axes(axes, numpy.eye(3))
    elif gpname in ('O',):
        return 'D2', align_axes(axes, numpy.eye(3))
    elif gpname in ('Ih',):
        return 'Ci', align_axes(axes, numpy.eye(3))
    elif gpname in ('I',):
        return 'C1', axes
    elif gpname in ('Td', 'T', 'Th'):
        #x,y,z = axes
        #x = _normalize(x+y)
        #y = numpy.cross(z, x)
        #return 'C2v', numpy.array((x,y,z))
        return 'D2', align_axes(axes, numpy.eye(3))
    elif re.search(r'S\d+', gpname):
        n = int(re.search(r'\d+', gpname).group(0))
        if n % 2 == 0:
            return 'C%d'%(n//2), axes
        else:
            return 'Ci', axes
    else:
        n = int(re.search(r'\d+', gpname).group(0))
        if n % 2 == 0:
            if re.search(r'D\d+d', gpname):
                subname = 'D2'
            elif re.search(r'D\d+h', gpname):
                subname = 'D2h'
            elif re.search(r'D\d+', gpname):
                subname = 'D2'
            elif re.search(r'C\d+h', gpname):
                subname = 'C2h'
            elif re.search(r'C\d+v', gpname):
                subname = 'C2v'
            else:
                subname = 'C2'
        else:
            # rotate axes and
            # Dnh -> C2v
            # Dn  -> C2
            # Dnd -> Ci
            # Cnh -> Cs
            # Cnv -> Cs
            if re.search(r'D\d+h', gpname):
                subname = 'C2v'
                axes = axes[[1,2,0]]
            elif re.search(r'D\d+d', gpname):
                subname = 'C2h'
                axes = axes[[1,2,0]]
            elif re.search(r'D\d+', gpname):
                subname = 'C2'
                axes = axes[[1,2,0]]
            elif re.search(r'C\d+h', gpname):
                subname = 'Cs'
            elif re.search(r'C\d+v', gpname):
                subname = 'Cs'
                axes = axes[[1,2,0]]
            else:
                subname = 'C1'
        return subname, axes


def symm_ops(gpname, axes=None):
    # FIXME: How does gpname enter into this?
    if axes is not None:
        raise RuntimeError('TODO: non-standard orientation')
    op1 = numpy.eye(3)
    opi = -1

    opc2z = -numpy.eye(3)
    opc2z[2,2] = 1
    opc2x = -numpy.eye(3)
    opc2x[0,0] = 1
    opc2y = -numpy.eye(3)
    opc2y[1,1] = 1

    opcsz = numpy.dot(opc2z, opi)
    opcsx = numpy.dot(opc2x, opi)
    opcsy = numpy.dot(opc2y, opi)
    opdic = {'E'  : op1,
             'C2z': opc2z,
             'C2x': opc2x,
             'C2y': opc2y,
             'i'  : opi,
             'sz' : opcsz,
             'sx' : opcsx,
             'sy' : opcsy,}
    return opdic

def symm_identical_atoms(gpname, atoms):
    ''' Requires '''
    # Dooh Coov for linear molecule
    if gpname == 'Dooh':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        idx0 = argsort_coords(coords)
        coords0 = coords[idx0]
        opdic = symm_ops(gpname)
        newc = numpy.dot(coords, opdic['sz'])
        idx1 = argsort_coords(newc)
        dup_atom_ids = numpy.sort((idx0,idx1), axis=0).T
        uniq_idx = numpy.unique(dup_atom_ids[:,0], return_index=True)[1]
        eql_atom_ids = dup_atom_ids[uniq_idx]
        eql_atom_ids = [list(sorted(set(i))) for i in eql_atom_ids]
        return eql_atom_ids
    elif gpname == 'Coov':
        eql_atom_ids = [[i] for i,a in enumerate(atoms)]
        return eql_atom_ids

    center = mole.charge_center(atoms)
#    if not numpy.allclose(center, 0, atol=TOLERANCE):
#        sys.stderr.write('WARN: Molecular charge center %s is not on (0,0,0)\n'
#                        % center)
    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    coords = numpy.array([a[1] for a in atoms], dtype=float)
    idx = argsort_coords(coords)
    coords0 = coords[idx]

    dup_atom_ids = []
    for op in ops:
        newc = numpy.dot(coords, op)
        idx = argsort_coords(newc)
        if not numpy.allclose(coords0, newc[idx], atol=TOLERANCE):
            raise RuntimeError('Symmetry identical atoms not found')
        dup_atom_ids.append(idx)

    dup_atom_ids = numpy.sort(dup_atom_ids, axis=0).T
    uniq_idx = numpy.unique(dup_atom_ids[:,0], return_index=True)[1]
    eql_atom_ids = dup_atom_ids[uniq_idx]
    eql_atom_ids = [list(sorted(set(i))) for i in eql_atom_ids]
    return eql_atom_ids

def check_given_symm(gpname, atoms, basis=None):
# more strict than symm_identical_atoms, we required not only the coordinates
# match, but also the symbols and basis functions

#FIXME: compare the basis set when basis is given
    if gpname == 'Dooh':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        if numpy.allclose(coords[:,:2], 0, atol=TOLERANCE):
            opdic = symm_ops(gpname)
            rawsys = SymmSys(atoms, basis)
            return rawsys.has_icenter() and numpy.allclose(rawsys.charge_center, 0)
        else:
            return False
    elif gpname == 'Coov':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        return numpy.allclose(coords[:,:2], 0, atol=TOLERANCE)

    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    rawsys = SymmSys(atoms, basis)
    for lst in rawsys.atomtypes.values():
        coords = rawsys.atoms[lst,1:]
        idx = argsort_coords(coords)
        coords0 = coords[idx]

        for op in ops:
            newc = numpy.dot(coords, op)
            idx = argsort_coords(newc)
            if not numpy.allclose(coords0, newc[idx], atol=TOLERANCE):
                return False
    return True

def shift_atom(atoms, orig, axis):
    c = numpy.array([a[1] for a in atoms])
    c = numpy.dot(c - orig, numpy.array(axis).T)
    return [[atoms[i][0], c[i]] for i in range(len(atoms))]

class RotationAxisNotFound(Exception):
    pass

class SymmSys(object):
    def __init__(self, atoms, basis=None, recenter_coords=True):
        self.atomtypes = mole.atom_types(atoms, basis)
        # fake systems, which treates the atoms of different basis as different atoms.
        # the fake systems do not have the same symmetry as the potential
        # it's only used to determine the main (Z-)axis
        chg1 = numpy.pi - 2 # A unique charge to label "decorated atoms"
                            # like H1 and O1 instead of H and O
        coords = []
        fake_chgs = []
        idx = []
        for k, lst in self.atomtypes.items():
            #print "K, lst = ", k, lst
            idx.append(lst)
            coords.append([atoms[i][1] for i in lst])
            ksymb = mole._rm_digit(k)
            if ksymb != k:
                # Put random charges on the decorated atoms
                fake_chgs.append([chg1] * len(lst))
                chg1 *= numpy.pi - 2
            elif 'GHOST' in ksymb:
                ksymb = mole._remove_prefix_ghost(ksymb)
                fake_chgs.append([mole._charge(ksymb)+.3] * len(lst))
            else:
                fake_chgs.append([mole._charge(ksymb)] * len(lst))
        coords = numpy.array(numpy.vstack(coords), dtype=float)
        fake_chgs = numpy.hstack(fake_chgs)
        self.charge_center = numpy.einsum('i,ij->j', fake_chgs, coords)/fake_chgs.sum()
        if recenter_coords:
            coords = coords - self.charge_center

        idx = numpy.argsort(numpy.hstack(idx))
        self.atoms = numpy.hstack((fake_chgs.reshape(-1,1), coords))[idx]

        self.group_atoms_by_distance = []
        decimals = int(-numpy.log10(TOLERANCE)) - 1
        for index in self.atomtypes.values():
            index = numpy.asarray(index)
            c = self.atoms[index,1:]
            dists = numpy.around(norm(c, axis=1), decimals)
            u, idx = numpy.unique(dists, return_inverse=True)
            for i, s in enumerate(u):
                self.group_atoms_by_distance.append(index[idx == i])

    def cartesian_tensor(self, n):
        """

        Provide different `n`-moment information for an atomic system.

        For example, dipole, Quadrupole, and Octopole correspond to `n = 1, 2, 3`,
        respectively.

        Parameters
        ----------
        n: int
            Angular momentum describing order of moment.

        Returns
        -------
        e: (M) ndarray
            Eigenvalues of moment matrix.
        w: (M, M) ndarray
            Eigenvectors of moment matrix, where the i'th eigenvector is given by w[:,i].

        """
        charge = self.atoms[:,0]
        coord = self.atoms[:,1:]
        ncart = (n+1)*(n+2)//2
        natom = len(charge)
        # Create charge array, normalized for good-behavior at higher orders
        tensor = numpy.sqrt(numpy.copy(charge).reshape(natom,-1) / charge.sum())
        for i in range(n):
            tensor = numpy.einsum('zi,zj->zij', tensor, coord).reshape(natom,-1)
        e, w = scipy.linalg.eigh(numpy.dot(tensor.T,tensor))
        return e[-ncart:], w[:,-ncart:]

    def invariant_to_op(self, op):
        for idx in self.group_atoms_by_distance:
            coords = self.atoms[idx, 1:]
            new_coords = numpy.dot(coords, op)
# FIXME: compare whehter two sets of coordinates are identical
            yield all((_vec_in_vecs(x, coords) for x in new_coords))

    def has_icenter(self):
        return all(self.invariant_to_op(-1))

    def has_rotation(self, axis, n):
        op = rotation_mat(axis, numpy.pi*2/n)
        return all(self.invariant_to_op(op))

    def has_mirror(self, perp_vec):
        '''

        Determines whether system is invariant to a mirror image across
        a hyperplane described by `perp_vec`.

        Parameters
        ----------
            perp_vec: (3,) array_like
                Vector describing hyperplane.

        Returns
        -------
            has_symmetry: bool
                Whether the mirror image preserves the system.

        '''
        return all(self.invariant_to_op(householder(perp_vec)))

    def search_possible_rotations(self, zaxis=None):
        '''

        Search for possible rotations in atomic system.

        Parameters
        ----------
            zaxis: (3,) array-like, optional
                Rotations must be about this specified z-axis.

        Returns
        -------
            rotation_and_order: list of ((3,) ndarray, int) tuples
                Each element of the list is a possible Cn rotation axis
                followed by the order of the rotation.

        '''
        maybe_cn = []
        for lst in self.group_atoms_by_distance:
            natm = len(lst)
            if natm > 1:
                coords = self.atoms[lst,1:]
# possible C2 axis
                for i in range(1, natm):
                    if abs(coords[0]+coords[i]).sum() > TOLERANCE:
                        maybe_cn.append((coords[0]+coords[i], 2))
                    else: # abs(coords[0]-coords[i]).sum() > TOLERANCE:
                        maybe_cn.append((coords[0]-coords[i], 2))

# atoms of equal distances may be associated with rotation axis > C2.
                r0 = coords - coords[0]
                distance = norm(r0, axis=1)
                eq_distance = abs(distance[:,None] - distance) < TOLERANCE
                for i in range(2, natm):
                    for j in numpy.where(eq_distance[i,:i])[0]:
                        cos = numpy.dot(r0[i],r0[j]) / (distance[i]*distance[j])
                        ang = numpy.arccos(cos)
                        nfrac = numpy.pi*2 / (numpy.pi-ang)
                        n = int(numpy.around(nfrac))
                        if abs(nfrac-n) < TOLERANCE:
                            maybe_cn.append((numpy.cross(r0[i],r0[j]),n))

        if maybe_cn == []:
            return None

        # remove zero-vectors and duplicated vectors
        vecs = numpy.vstack([x[0] for x in maybe_cn])
        idx = norm(vecs, axis=1) > TOLERANCE
        ns = numpy.hstack([x[1] for x in maybe_cn])
        vecs = _normalize(vecs[idx])
        ns = ns[idx]

        if zaxis is not None:  # Only keep parallel rotation axes to zaxis
            cos = numpy.dot(vecs, _normalize(zaxis))
            vecs = vecs[(abs(cos-1) < TOLERANCE) | (abs(cos+1) < TOLERANCE)]
            ns = ns[(abs(cos-1) < TOLERANCE) | (abs(cos+1) < TOLERANCE)]

        possible_cn = []
        seen = numpy.zeros(len(vecs), dtype=bool)
        for k, v in enumerate(vecs):
            if not seen[k]:
                where1 = numpy.einsum('ix->i', abs(vecs[k:] - v)) < TOLERANCE
                where1 = numpy.where(where1)[0] + k
                where2 = numpy.einsum('ix->i', abs(vecs[k:] + v)) < TOLERANCE
                where2 = numpy.where(where2)[0] + k
                seen[where1] = True
                seen[where2] = True

                vk = _normalize((numpy.einsum('ix->x', vecs[where1]) -
                                 numpy.einsum('ix->x', vecs[where2])))
                for n in (set(ns[where1]) | set(ns[where2])):
                    possible_cn.append((vk,n))
        return possible_cn

    def search_c2x(self, zaxis, n):
        '''C2 axis which is perpendicular to z-axis'''
        decimals = int(-numpy.log10(TOLERANCE)) - 1
        for lst in self.group_atoms_by_distance:
            if len(lst) > 1:
                r0 = self.atoms[lst,1:]
                zcos = numpy.around(numpy.einsum('ij,j->i', r0, zaxis),
                                    decimals=decimals)
                uniq_zcos = numpy.unique(zcos)
                maybe_c2x = []
                for d in uniq_zcos:
                    if d > TOLERANCE:
                        mirrord = abs(zcos+d)<TOLERANCE
                        if mirrord.sum() == (zcos==d).sum():
                            above = r0[zcos==d]
                            below = r0[mirrord]
                            nelem = len(below)
                            maybe_c2x.extend([above[0] + below[i]
                                              for i in range(nelem)])
                    elif abs(d) < TOLERANCE: # plane which crosses the orig
                        r1 = r0[zcos==d][0]
                        maybe_c2x.append(r1)
                        r2 = numpy.dot(r1, rotation_mat(zaxis, numpy.pi*2/n))
                        if abs(r1+r2).sum() > TOLERANCE:
                            maybe_c2x.append(r1+r2)
                        else:
                            maybe_c2x.append(r2-r1)

                if len(maybe_c2x) > 0:
                    idx = norm(maybe_c2x, axis=1) > TOLERANCE
                    maybe_c2x = _normalize(maybe_c2x)[idx]
                    maybe_c2x = _remove_dupvec(maybe_c2x)
                    for c2x in maybe_c2x:
                        if (not parallel_vectors(c2x, zaxis) and
                            self.has_rotation(c2x, 2)):
                            return c2x

    def search_mirrorx(self, n, zaxis=None):
        '''mirror which is parallel to z-axis'''
        # If n > 1 is given, we expect a rotation axis to be given
        if n > 1 and zaxis is None:
            raise ValueError('For a non-trivial Cn axis of', n, 'an axis of '
                             'rotation is expected.  Input zaxis is ', zaxis)
        if n > 1:
            for lst in self.group_atoms_by_distance:
                natm = len(lst)
                r0 = self.atoms[lst[0],1:]
                if natm > 1 and not parallel_vectors(r0, zaxis):
                    r1 = numpy.dot(r0, rotation_mat(zaxis, numpy.pi*2/n))
                    mirrorx = _normalize(r1-r0)
                    if self.has_mirror(mirrorx):
                        return mirrorx
        else:
            # Search for all vectors between two atoms for mirror symmetry
            for lst in self.group_atoms_by_distance:
                natm = len(lst)
                r0 = self.atoms[lst,1:]
                if natm > 1:
                    maybe_mirror = [r0[i]-r0[0] for i in range(1, natm)]
                    for mirror in _normalize(maybe_mirror):
                        if self.has_mirror(mirror):
                            return mirror

    def search_highest_cn(self, zaxis=None):
        '''

        Searches for the highest order Cn rotation axis.

        Parameters
        ----------
            zaxis: (3,) array-like, optional
                Rotations must be about this specified z-axis.

        Returns
        -------
            cn_axis: (3,) ndarray
                Highest order Cn rotation axis.
            nmax: int
                Order of highest order Cn rotation axis.

        '''
        possible_cn = self.search_possible_rotations(zaxis)
        if possible_cn is None:
            return None, None
        nmax = 1
        cmax = numpy.array([0.,0.,1.])
        # Loop through possible rotations for highest order Cn,
        # checking if the rotation actually preserves the system.
        for cn, n in possible_cn:
            if n > nmax and self.has_rotation(cn, n):
                nmax = n
                cmax = cn
        return cmax, nmax


def _normalize(vecs):
    vecs = numpy.asarray(vecs)
    if vecs.ndim == 1:
        return vecs / (numpy.linalg.norm(vecs) + 1e-200)
    else:
        return vecs / (norm(vecs, axis=1).reshape(-1,1) + 1e-200)

def _vec_in_vecs(vec, vecs):
    norm = numpy.sqrt(len(vecs))
    return min(numpy.einsum('ix->i', abs(vecs-vec))/norm) < TOLERANCE

def _search_i_group(rawsys):
    possible_cn = rawsys.search_possible_rotations()
    c5_axes = [c5 for c5, n in possible_cn
               if n == 5 and rawsys.has_rotation(c5, 5)]
    if len(c5_axes) <= 1:
        return None,None

    zaxis = c5_axes[0]
    cos = numpy.dot(c5_axes, zaxis)
    assert(numpy.all((abs(cos[1:]+1/numpy.sqrt(5)) < TOLERANCE) |
                     (abs(cos[1:]-1/numpy.sqrt(5)) < TOLERANCE)))

    if rawsys.has_icenter():
        gpname = 'Ih'
    else:
        gpname = 'I'

    c5 = c5_axes[1]
    if numpy.dot(c5, zaxis) < 0:
        c5 = -c5
    c5a = numpy.dot(zaxis, rotation_mat(zaxis, numpy.pi*6/5))
    xaxis = c5a + c5
    return gpname, _make_axes_from_2d(zaxis, xaxis)

def _search_ot_group(rawsys):
    possible_cn = rawsys.search_possible_rotations()
    c4_axes = [c4 for c4, n in possible_cn
               if n == 4 and rawsys.has_rotation(c4, 4)]

    if len(c4_axes) > 0:  # T group
        assert(len(c4_axes) > 1)
        if rawsys.has_icenter():
            gpname = 'Oh'
        else:
            gpname = 'O'
        return gpname, _make_axes_from_2d(c4_axes[0], c4_axes[1])

    else:  # T group
        c3_axes = [c3 for c3, n in possible_cn
                   if n == 3 and rawsys.has_rotation(c3, 3)]
        if len(c3_axes) <= 1:
            return None, None

        cos = numpy.dot(c3_axes, c3_axes[0])
        assert(numpy.all((abs(cos[1:]+1./3) < TOLERANCE) |
                         (abs(cos[1:]-1./3) < TOLERANCE)))

        if rawsys.has_icenter():
            gpname = 'Th'
# Because C3 axes are on the mirror of Td, two C3 can determine a mirror.
        elif rawsys.has_mirror(numpy.cross(c3_axes[0], c3_axes[1])):
            gpname = 'Td'
        else:
            gpname = 'T'

        c3a = c3_axes[0]
        if numpy.dot(c3a, c3_axes[1]) > 0:
            c3a = -c3a
        c3b = numpy.dot(c3_axes[1], rotation_mat(c3a, numpy.pi*2/3))
        c3c = numpy.dot(c3_axes[1], rotation_mat(c3a,-numpy.pi*2/3))
        zaxis, xaxis = c3a+c3b, c3a+c3c
        return gpname, _make_axes_from_2d(zaxis, xaxis)

def _degeneracy(e, decimals):
    e = numpy.around(e, decimals)
    u, idx = numpy.unique(e, return_inverse=True)
    degen = [numpy.count_nonzero(idx==i) for i in range(len(u))]
    return degen

def _pesudo_vectors(vs):
    idy0 = abs(vs[:,1])<TOLERANCE
    idz0 = abs(vs[:,2])<TOLERANCE
    vs = vs.copy()
    # ensure z component > 0
    vs[vs[:,2]<0] *= -1
    # if z component == 0, ensure y component > 0
    vs[(vs[:,1]<0) & idz0] *= -1
    # if y and z component == 0, ensure x component > 0
    vs[(vs[:,0]<0) & idy0 & idz0] *= -1
    return vs

def _remove_dupvec(vs):
    def rm_iter(vs):
        if len(vs) <= 1:
            return vs
        else:
            x = numpy.sum(abs(vs[1:]-vs[0]), axis=1)
            rest = rm_iter(vs[1:][x>TOLERANCE])
            return numpy.vstack((vs[0], rest))
    return rm_iter(_pesudo_vectors(vs))

def _make_axes_from_2d(z, x):
    '''

    Makes a set of 3D cartesian axes from two axes.

    Parameters
    ----------
        z: (3,) array_like
            Cartesian z-axis.
        x: (3,) array_like
            Cartesian x-axis.

    Notes
    -----
        The z-axis will remain the same, but x may not
        depending on if z, x were orthogonal.

    '''
    y = numpy.cross(z, x)
    x = numpy.cross(y, z) # because x might not be perp to z
    return _normalize(numpy.array((x,y,z)))

def _refine(axes):
# Make sure the axes can be rotated from continuous unitary transformation
    if axes[2,2] < 0:
        axes[2] *= -1
    if abs(axes[0,0]) > abs(axes[1,0]):
        x_id, y_id = 0, 1
    else:
        x_id, y_id = 1, 0
    if axes[x_id,0] < 0:
        axes[x_id] *= -1
    if numpy.linalg.det(axes) < 0:
        axes[y_id] *= -1
    return axes


if __name__ == "__main__":
    atom = [["O" , (1. , 0.    , 0.   ,)],
            ['H' , (0. , -.757 , 0.587,)],
            ['H' , (0. , 0.757 , 0.587,)] ]
    gpname, orig, axes = detect_symm(atom)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (1,0,0)], ['H', (1,0,-1)], ['H', (1,0,1)]]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, orig, axes)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (0., 0., 0.)],
            ['H', (0., 0., 1.)],
            ['H', (0., 1., 0.)],
            ['H', (1., 0., 0.)],
            ['H', (-1, 0., 0.)],
            ['H', (0.,-1., 0.)],
            ['H', (0., 0.,-1.)]]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, orig, axes)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(subgroup(gpname, axes)[0], atom))
