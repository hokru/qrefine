from __future__ import division

import os
import ase.units as ase_units
import mmtbx.restraints
from libtbx.utils import Sorry
from charges import charges_class
from scitbx.array_family import flex
from clustering import betweenness_centrality_clustering
from plugin.ase.mopac_qr import Mopac
from plugin.ase.pyscf_qr import Pyscf
from plugin.ase.terachem_qr import TeraChem
from plugin.ase.turbomole_qr import Turbomole
from plugin.ase.orca_qr import Orca
from plugin.ase.gaussian_qr import Gaussian
from libtbx import group_args

class from_cctbx(object):
  def __init__(self, restraints_manager, fragment_extracts=None,
              file_name="./ase/tmp_ase.pdb"):
    self.geometry_restraints_manager = restraints_manager
    self.file_name = file_name
    self.fragment_extracts = fragment_extracts

  def __call__(self, selection_and_sites_cart):
    return self.target_and_gradients(
      sites_cart = selection_and_sites_cart[1],
      selection  = selection_and_sites_cart[0],
      index      = selection_and_sites_cart[2])
      
  def energies_sites(self, sites_cart, compute_gradients=True):
    tg = self.target_and_gradients(sites_cart=sites_cart)
    return group_args(
      target    = tg[0],
      gradients = tg[1])

  def target_and_gradients(self, sites_cart, selection=None, index=None):
    if(selection is not None): ### clustering
      super_selection = self.fragment_extracts.fragment_super_selections[index]
      grm = self.fragment_extracts.super_sphere_geometry_restraints_manager
      es = grm.select(super_selection).energies_sites(
        sites_cart=sites_cart.select(super_selection), compute_gradients=True)
      es.gradients = es.gradients[:selection.count(True)]
      es.gradients = es.gradients * flex.double(
            self.fragment_extracts.fragment_scales[index])
    else:
      es = self.geometry_restraints_manager.energies_sites(
        sites_cart=sites_cart, compute_gradients=True)
    return es.target, es.gradients

class from_qm(object):
  def __init__(self,
      fragment_extracts          = None,
      pdb_hierarchy              = None,
      charge                     = None,
      qm_engine_name             = None,
      file_name                  = "./ase/tmp_ase.pdb",
      crystal_symmetry           = None,
      clustering                 = False,
#      charge_service             = None,
      cif_objects                = None,
      # change to quantum phil scope !!!!
      method                     = 'rhf',
      basis                      = "sto-3g",
      memory                     = None,
      nproc                      = None,
  ):
    self.fragment_extracts  = fragment_extracts
    self.method = method
    self.basis = basis
    self.memory = memory
    self.nproc = nproc

    self.pdb_hierarchy = pdb_hierarchy
    self.qm_engine_name = qm_engine_name
    self.file_name = file_name
    self.working_folder = os.path.split(self.file_name)[0]+ "/"
    if(os.path.exists(self.working_folder) is not True):
      os.mkdir(self.working_folder)
    if(charge is None and clustering is False):
      #raw_records = pdb_hierarchy.as_pdb_string(crystal_symmetry=crystal_symmetry)
      #cc = charges_class(raw_records=raw_records)
      #self.charge = cc.get_total_charge()
      #@Nigel
      raw_records = pdb_hierarchy.as_pdb_string(crystal_symmetry=crystal_symmetry)
      charge_service = charges_class(raw_records = raw_records,
                                     cif_objects = cif_objects)
      self.charge = charge_service.get_total_charge()
    else: self.charge = charge
    self.clustering = clustering
    self.qm_engine = self.create_qm_engine()
    self.qm_engine.command = self.qm_engine.get_command()
    self.system_size = self.pdb_hierarchy.atoms_size()

  def create_qm_engine(self):
    if(self.qm_engine_name == "turbomole"):
      calculator = Turbomole()
    elif(self.qm_engine_name == "terachem"):
      ### if TeraChem has problem reading pdb file, update TeraChem version.
      calculator = TeraChem(gpus="4",
                            basis=self.basis,
                            dftd="yes",
                            watcheindiis="yes",
                            scf="diis+a")
    elif(self.qm_engine_name == "mopac"):
      calculator = Mopac()
    elif(self.qm_engine_name == "pyscf"):
      calculator = Pyscf()
    elif(self.qm_engine_name == "orca"):
      calculator = Orca()
    elif(self.qm_engine_name == "gaussian"):
      calculator = Gaussian()
    else:
      raise Sorry("qm_calculator needs to be specified.")
    #
    # set to appropriate values
    #
    for attr in ['charge',
                 'basis',
                 'method',
                 'memory',
                 'nproc',
                 ]:
      value = getattr(self, attr, None)
      func = getattr(calculator, 'set_%s' % attr, None)
      action=False
      if func is not None:
        if value is not None:
          #print '  Setting %s to %s' % (attr, value)
          func(value)
          action=True
      if not action:
        if value and not func:
          print '  No function available to set %s to %s' % (attr, value)
    return calculator

  def __call__(self,fragment_selection_and_sites_cart):
    return self.target_and_gradients(
      sites_cart = fragment_selection_and_sites_cart[1],
      selection  = fragment_selection_and_sites_cart[0],
      index      = fragment_selection_and_sites_cart[2])
      
  def energies_sites(self, sites_cart, compute_gradients=True):
    tg = self.target_and_gradients(sites_cart=sites_cart)
    return group_args(
      target    = tg[0],
      gradients = tg[1])

  def target_and_gradients(self,sites_cart, selection=None, index=None):
    if(self.clustering):
      from fragment import get_qm_file_name_and_pdb_hierarchy
      from fragment import charge
      from fragment import write_mm_charge_file
      #
      qm_pdb_file, ph = get_qm_file_name_and_pdb_hierarchy(
                          fragment_extracts=self.fragment_extracts,
                          index=index)
      #
      qm_charge = charge(fragment_extracts=self.fragment_extracts,
                                      index=index)
      charge_file =  write_mm_charge_file(fragment_extracts=self.fragment_extracts,
                                      index=index)
      gradients_scale = self.fragment_extracts.fragment_scales[index]
    else:
      self.pdb_hierarchy.atoms().set_xyz(sites_cart)
      self.pdb_hierarchy.write_pdb_file(file_name=self.file_name)
      ph = self.pdb_hierarchy## return pdb_hierarchy
      qm_pdb_file = self.file_name
      qm_charge = self.charge
      charge_file = None
      selection =flex.bool(self.system_size, True)
      gradients_scale = [1.0]*self.system_size
    #
    # need to get commands for QM programs from local machine so queue machines
    # know where to look.
    #
    define_str=''
    command = None
    if (self.qm_engine_name == 'terachem'):
       print "terachem is not updated to the new API"
    elif (self.qm_engine_name == 'mopac'):
      command = self.qm_engine.get_command()
    elif (self.qm_engine_name == 'gaussian'):
      command = self.qm_engine.get_command()
    elif (self.qm_engine_name == 'orca'):
      command = self.qm_engine.get_command()
    elif (self.qm_engine_name == 'turbomole'):
      command = self.qm_engine.get_command()
    else:
      assert 0
    atoms = ase_atoms_from_pdb_hierarchy(ph)
    self.qm_engine.set_label(qm_pdb_file[:-4])
    self.qm_engine.run_qr(atoms,
                          charge=qm_charge,
                          pointcharges=charge_file,
                          coordinates=qm_pdb_file[:-4]+".xyz",
                          command=command,       # 
                          define_str=define_str, # for Terachem
      )
    unit_convert = ase_units.mol/ase_units.kcal
    energy = self.qm_engine.energy_free*unit_convert
    ase_gradients = (-1.0) * self.qm_engine.forces*unit_convert
    # remove capping and neibouring buffer
    gradients = ase_gradients[:selection.count(True)]
    gradients =  flex.vec3_double(gradients)
    ## TODO
    ## unchange the altloc gradient, averagely scale the non-altloc gradient
    gradients = gradients*flex.double(gradients_scale)
    return energy, gradients

from ase import Atoms
def ase_atoms_from_pdb_hierarchy(ph):

  def read_pdb_hierarchy(pdb_hierarchy):
    positions = []
    symbols = []
    for chain in pdb_hierarchy.chains():
      for residue_group in chain.residue_groups():
        for atom in residue_group.atoms():
          element = atom.element.strip()
          if (len(element) == 2):
            element = element[0] + element[1].lower()
          symbols.append(element)
          positions.append(list(atom.xyz))
    return symbols, positions

  symbols, positions = read_pdb_hierarchy(ph)
  return Atoms(symbols=symbols, positions=positions)
