"""
ASE plugin for DFTB+ (http://www.dftbplus.org/).

Based on the xtb/mopac Q|R plugins. Unlike ASE's own
ase.calculators.dftb.Dftb (a FileIOCalculator with a fixed, single-string
'command'), this plugin builds an arbitrarily long mpirun/mpiexec command.

Default Hamiltonian is selected with the method string ``gfn2/alpb``.
Supported combinations are GFN1 or GFN2 with ALPB, GBSA, or no solvation.
They need no external parameter or Slater-Koster parameter files. DFTB+'s
ALPB input is configured to reproduce tblite's CLI result within the SCC
convergence tolerance. Classic SCC-DFTB with
Slater-Koster (.skf) files is not wired up here -- conda-forge's dftbplus
package does not ship any .skf sets; users who need them must obtain a set
(e.g. 3ob, mio from dftb.org) themselves and pass a custom
hamiltonian_params dict selecting Hamiltonian_ = 'DFTB' with the appropriate
SlaterKosterFiles block.
"""
import copy
import os
import sys

import numpy as np
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.units import Bohr, Hartree, kcal, mol

key_parameters = {
                  'coordinates': None,
                  'charge': None,
                  'method': None,
                  'pointcharges': None,
                  'nproc': 1,
                  }

_DEFAULT_METHOD = 'gfn2/alpb'
_VALID_METHODS = tuple(
    '%s/%s' % (gfn, solvation)
    for gfn in ('gfn1', 'gfn2')
    for solvation in ('alpb', 'gbsa', 'none'))


def normalize_dftb_method(method):
    """Validate and normalize Q|R's compact DFTB+ method syntax."""
    if not isinstance(method, str):
        raise ValueError(
            'Invalid DFTB+ quantum.method %r. Choose one of: %s.' %
            (method, ', '.join(_VALID_METHODS)))
    normalized = method.strip().lower()
    if normalized not in _VALID_METHODS:
        raise ValueError(
            'Invalid DFTB+ quantum.method %r. Choose one of: %s.' %
            (method, ', '.join(_VALID_METHODS)))
    return normalized


def hamiltonian_params_for_method(method):
    """Return DFTB+ HSD parameters for a validated Q|R method string."""
    method = normalize_dftb_method(method)
    gfn, solvation = method.split('/')
    params = {
        'Hamiltonian_': 'xTB',
        'Hamiltonian_Method': '"%s-xTB"' % gfn.upper(),
        'Options_': '',
        'Options_WriteResultsTag': 'Yes',
    }
    if solvation == 'none':
        return params

    params.update({
        # DFTB+'s Solvation input is independent of its xTB/tblite interface.
        # These select the same practical ALPB setup as `tblite --method gfn2
        # --alpb water`: COSMO radii, P16 kernel, unit Born scale, no Born
        # offset, and a 0.8 descreening factor for every species (added at
        # write time). The DFTB+ native implementation then agrees with tblite
        # within the respective SCC convergence tolerance.
        'Hamiltonian_Solvation_': 'GeneralisedBorn',
        'Hamiltonian_Solvation_ALPB': 'Yes' if solvation == 'alpb' else 'No',
        'Hamiltonian_Solvation_BornOffset': '0.0',
        'Hamiltonian_Solvation_BornScale': '1.0',
        'Hamiltonian_Solvation_Descreening_': 'Values',
        'Hamiltonian_Solvation_FreeEnergyShift': '0.0',
        'Hamiltonian_Solvation_Kernel': '"P16"',
        'Hamiltonian_Solvation_Radii_': 'vanDerWaalsRadiiCosmo',
        'Hamiltonian_Solvation_Solvent_': 'fromName',
        'Hamiltonian_Solvation_Solvent__empty': '"water"',
    })
    return params


default_hamiltonian_params = hamiltonian_params_for_method(_DEFAULT_METHOD)


class DFTBPlus(Calculator):
    name = 'dftb+'

    def __init__(self,
                 label='ase_plugin',
                 coordinates='geo_end.gen',
                 charge='0',
                 method=None,
                 nproc=1,
                 atoms=None,
                 pointcharges=None,
                 hamiltonian_params=None,
                 extra_flags='',
                 **kwargs):

        self.coordinates = coordinates
        self.pointcharges = pointcharges
        self.key_parameters = copy.deepcopy(key_parameters)
        self.key_parameters['coordinates'] = coordinates
        self.key_parameters['charge'] = charge
        self.key_parameters['method'] = None
        self.key_parameters['nproc'] = nproc
        self.extra_flags = extra_flags
        self._uses_managed_hamiltonian = hamiltonian_params is None
        if hamiltonian_params is None:
            method = _DEFAULT_METHOD if method is None else method
            method = normalize_dftb_method(method)
            hamiltonian_params = hamiltonian_params_for_method(method)
            self.key_parameters['method'] = method
        else:
            self.key_parameters['method'] = method
        self.hamiltonian_params = hamiltonian_params
        # save label
        self.label = label
        # set atoms
        self.atoms = atoms
        # initialize the results
        self.energy_zero = None
        self.energy_free = None
        self.forces = None
        self.stress = None
        self.command = self.get_command()
        self.calc_dir = None

    def get_command(self):
        """Return command string for the dftb+ binary if installed, else None.
           Search order: 1. DFTB_COMMAND 2. $PATH 3. pixi/conda env bin """
        from shutil import which
        if 'DFTB_COMMAND' in os.environ:
            return os.environ['DFTB_COMMAND']
        command = which('dftb+')
        if command is not None:
            return command
        command = os.path.join(sys.prefix, 'bin', 'dftb+')
        if os.path.exists(command):
            return command
        return None

    def get_mpi_prefix(self):
        """Build the mpirun/mpiexec prefix used to launch dftb+.
           Kept separate from self.command so nproc/extra flags can be set
           individually via the Q|R set_* protocol without needing to
           re-assemble a full command string by hand.

           MPI-vendor-agnostic by design: only '-np N' is emitted, since
           the flag syntax for everything else (env-var forwarding,
           hostfiles, binding, ...) differs across MPICH/Open MPI/vendor
           MPIs and there's no portable spelling. OMP_NUM_THREADS=1 (needed
           because ELSI's eigensolvers don't support OpenMP threading) is
           set in run_command()'s subprocess environment instead of via an
           mpirun flag -- mpirun forwards the launching process's
           environment to ranks by default for every MPI implementation
           tested, so no vendor-specific flag is needed for that case. If a
           user's launcher/setup needs explicit env forwarding (e.g. across
           hosts without a shared environment) or any other mpirun option,
           they must supply it themselves via extra_flags -- this
           plugin does not guess vendor-specific flag syntax on the user's
           behalf."""
        nproc = int(self.key_parameters['nproc'])
        if nproc <= 1:
            return ''
        from shutil import which
        # Prefer the pixi/conda env's own mpirun over $PATH when present,
        # since an mpirun from a different MPI implementation than the one
        # dftb+ is linked against typically fails to launch it at all.
        mpirun = os.path.join(sys.prefix, 'bin', 'mpirun')
        if not os.path.exists(mpirun):
            mpirun = which('mpirun') or 'mpirun'
        prefix = '%s -np %i' % (mpirun, nproc)
        if self.extra_flags:
            prefix += ' ' + self.extra_flags
        return prefix

    def run_command(self, command):
        """
        execute <command> in a subprocess and check error code

        DFTB+'s MPI-linked binaries abort unless OpenMP threading is either
        explicitly enabled (UseOmpThreads = Yes in the Parallel block) or
        OMP_NUM_THREADS=1 is set -- this applies even when running without
        mpirun, since the binary itself is MPI-linked. ELSI's eigensolvers
        additionally don't support OpenMP parallelism, so forcing 1 thread
        here is also the documented-safe setting when ELSI/ELPA is used.
        """
        from subprocess import Popen, PIPE
        if command == '':
            raise RuntimeError('no command for run_command :(')
        print('Running: ', command)  # debug
        env = dict(os.environ)
        env.setdefault('OMP_NUM_THREADS', '1')
        proc = Popen([command], shell=True, stderr=PIPE, env=env)
        proc.wait()
        exitcode = proc.returncode
        if exitcode != 0:
            print(proc.communicate())
            error = '%s exited with error code %i in %s' % (
                           command, exitcode, self.calc_dir)
            raise RuntimeError(error)
        return 0

    def write_dftb_in(self, outfile):
        """Write dftb_in.hsd from self.hamiltonian_params, following the
           flattened-key convention of ase.calculators.dftb.Dftb
           (Hamiltonian_SCC='Yes' -> 'SCC = Yes' nested under Hamiltonian)."""
        outfile.write('Geometry = GenFormat { \n')
        outfile.write('    <<< "geo_end.gen" \n')
        outfile.write('} \n')
        outfile.write(' \n')

        params = dict(self.hamiltonian_params)
        if self._uses_managed_hamiltonian and any(
                key.startswith('Hamiltonian_Solvation_') for key in params):
            # DFTB+'s GeneralisedBorn reader requires a value per species,
            # while tblite supplies its default of 0.8 internally.
            for symbol in sorted(set(self.atoms.get_chemical_symbols())):
                params['Hamiltonian_Solvation_Descreening_' + symbol] = '0.8'
        charge = self.key_parameters.get('charge')
        if charge not in (None, '0', 0) and 'Hamiltonian_Charge' not in params:
            params['Hamiltonian_Charge'] = str(float(charge))

        open_blocks = []
        for key, value in sorted(params.items()):
            if key.endswith('_'):
                path = tuple(key.rstrip('_').split('_'))
                parent = path[:-1]
                is_block = True
            elif key.count('_empty') == 1:
                path = tuple(key.replace('_empty', '').rstrip('_').split('_'))
                parent = path
                is_block = False
            else:
                path = tuple(key.split('_'))
                parent = path[:-1]
                is_block = False

            # Flattened keys are sorted rather than strictly depth-first, so
            # track actual open HSD blocks instead of inferring their number
            # from underscores. This handles fromName { "water" }.
            common = 0
            for opened, wanted in zip(open_blocks, parent):
                if opened != wanted:
                    break
                common += 1
            while len(open_blocks) > common:
                outfile.write(3 * (len(open_blocks) - 1) * ' ' + '} \n')
                open_blocks.pop()
            if tuple(open_blocks) != parent:
                raise ValueError('invalid flattened HSD key order: %s' % key)

            outfile.write(3 * len(open_blocks) * ' ')
            if is_block and len(value) > 0:
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' = ' + str(value) + '{ \n')
            elif is_block and len(value) == 0 and len(parent) == 0:
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' ' + str(value) + '{ \n')
            elif is_block and len(value) == 0:
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' = ' + str(value) + '{ \n')
            elif key.count('_empty') == 1:
                outfile.write(str(value) + ' \n')
            else:
                outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
            if is_block:
                open_blocks.append(path[-1])
            if (is_block and self.pointcharges is not None
                    and ('xTB' in str(value) or 'DFTB' in str(value))):
                outfile.write('   ElectricField = { \n')
                outfile.write('      PointCharges = { \n')
                outfile.write(
                    '         CoordsAndCharges [Angstrom] = DirectRead { \n')
                outfile.write('            Records = ' +
                              str(self._n_pointcharges) + ' \n')
                outfile.write(
                    '            File = "dftb_external_charges.dat" \n')
                outfile.write('         } \n')
                outfile.write('      } \n')
                outfile.write('   } \n')
        while open_blocks:
            outfile.write(3 * (len(open_blocks) - 1) * ' ' + '} \n')
            open_blocks.pop()
        outfile.write('ParserOptions { \n')
        outfile.write('   IgnoreUnprocessedNodes = Yes  \n')
        outfile.write('} \n')
        outfile.write('Analysis { \n')
        outfile.write('   PrintForces = Yes  \n')
        outfile.write('} \n')

    def write_input(self, atoms):
        atoms = copy.deepcopy(self.atoms)
        write('geo_end.gen', atoms, format='gen', parallel=False)
        with open('dftb_in.hsd', 'w') as outfile:
            self.write_dftb_in(outfile)

    def write_pointcharges(self):
        """Write DFTB+'s external point-charge file: 'x y z charge' per
           line, in Angstrom (same format as ASE's PointChargePotential).
           Source file (from fragment.py's write_pdb_hierarchy_qxyz_file,
           same format xtb_qr.py consumes) is 'charge x y z' per line with
           a 2-line header (atom count, blank line)."""
        f = open(self.pointcharges, "r")
        pchrg = f.readlines()
        f.close()
        entries = pchrg[2:]
        self._n_pointcharges = len(entries)
        f = open("dftb_external_charges.dat", "w")
        for line in entries:
            words = line.split()
            q, x, y, z = (float(w) for w in words[:4])
            f.write('%12.6f %12.6f %12.6f %12.6f \n' % (x, y, z, q))
        f.close()

    def run_qr(self, atoms, coordinates, charge, pointcharges, **kwargs):
        """
        Handles DFTB+ calculations
        """
        self.atoms = atoms
        self.key_parameters['charge'] = charge

        working_dir = os.getcwd()
        if not os.path.isdir(self.label):
            os.mkdir(self.label)
        self.calc_dir = os.path.join(working_dir, self.label)
        os.chdir(self.calc_dir)

        self.pointcharges = pointcharges
        if self.pointcharges is not None:
            self.pointcharges = os.path.abspath(self.pointcharges)

        # write pointcharges file (if any) before dftb_in.hsd, since
        # write_dftb_in needs self._n_pointcharges for the ElectricField block
        if self.pointcharges is not None:
            self.write_pointcharges()

        self.write_input(self.atoms)

        binary = self.command
        if binary is None:
            raise RuntimeError('DFTB_COMMAND is not specified')

        mpi_prefix = self.get_mpi_prefix()
        if mpi_prefix:
            command = '%s %s > dftb.out' % (mpi_prefix, binary)
        else:
            command = '%s > dftb.out' % (binary,)

        # clean up
        for f in ['results.tag', 'detailed.out']:
            if os.path.exists(f):
                os.remove(f)

        self.run_command(command)
        self.read_results()
        self.energy_zero = self.energy_free
        os.chdir(working_dir)

    def read_results(self):
        """Read energy from detailed.out and forces from results.tag
           (both in Hartree/Bohr, DFTB+'s native atomic-unit output),
           converting to kcal/mol / (kcal/mol/Angstrom) as used elsewhere
           in Q|R."""
        with open('detailed.out', 'r') as fd:
            lines = fd.readlines()
        energy_hartree = None
        for line in lines:
            if line.strip().startswith('Total energy:'):
                energy_hartree = float(line.split()[2])
                break
        if energy_hartree is None:
            raise RuntimeError('DFTB+: could not find total energy in detailed.out')
        self.e_au = energy_hartree
        self.energy_free = energy_hartree * Hartree / (kcal / mol)

        with open('results.tag', 'r') as fd:
            tag_lines = fd.readlines()

        index_force_begin = -1
        index_force_end = -1
        for iline, line in enumerate(tag_lines):
            if line.find('forces   ') >= 0:
                index_force_begin = iline + 1
                line1 = line.replace(':', ',')
                index_force_end = iline + 1 + int(line1.split(',')[-1])
                break
        if index_force_begin < 0:
            raise RuntimeError('DFTB+: could not find forces in results.tag')

        forces_au = []
        for j in range(index_force_begin, index_force_end):
            word = tag_lines[j].split()
            forces_au.append([float(word[k]) for k in range(3)])
        # results.tag stores physical forces (not energy gradients), so
        # unlike xtb's turbomole-gradient parser, no sign flip is needed here.
        self.forces = np.array(forces_au) * (Hartree / Bohr) / (kcal / mol)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self.key_parameters[str(key)] = value

    # Q|R requirements
    def set_charge(self, charge):
        self.key_parameters['charge'] = charge

    def set_method(self, method):
        method = normalize_dftb_method(method)
        self.key_parameters['method'] = method
        if self._uses_managed_hamiltonian:
            self.hamiltonian_params = hamiltonian_params_for_method(method)

    def set_label(self, label):
        self.label = label

    def set_nproc(self, nproc):
        self.key_parameters['nproc'] = int(nproc)

    # DFTB+-specific extensions (harmless no-ops for other engines,
    # since Q|R's generic setter loop only calls setters that exist)
    def set_extra_flags(self, extra_flags):
        self.extra_flags = extra_flags

    def set_hamiltonian_params(self, hamiltonian_params):
        self._uses_managed_hamiltonian = False
        self.hamiltonian_params = hamiltonian_params
