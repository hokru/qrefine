from __future__ import division
# LIBTBX_SET_DISPATCHER_NAME qr.fragmentation
import sys
import time
import os.path
import libtbx
import iotbx.pdb
from qrefine.fragment import fragments
from qrefine.fragment import fragment_extracts
from qrefine.fragment import get_qm_file_name_and_pdb_hierarchy
from qrefine.fragment import charge
from qrefine.fragment import write_mm_charge_file

qrefine_path = libtbx.env.find_in_repositories("qrefine")
qr_yoink_path =os.path.join(qrefine_path, "plugin","yoink","yoink")

log = sys.stdout

legend = """\
Cluster a system into many small pieces
"""

def run(pdb_file, log):
  pdb_inp = iotbx.pdb.input(pdb_file)
  ph = pdb_inp.construct_hierarchy()
  cs = pdb_inp.crystal_symmetry()
  fq = fragments(
    pdb_hierarchy=ph,
    crystal_symmetry=cs,
    charge_embedding=True,
    debug=True,
    qm_engine_name="terachem")
  print >> log, "Residue indices for each cluster:\n", fq.clusters
  fq_ext = fragment_extracts(fq)
  for i in xrange(len(fq.clusters)):
      # add capping for the cluster and buffer
      print >> log, "capping frag:", i
      get_qm_file_name_and_pdb_hierarchy(
                          fragment_extracts=fq_ext,
                          index=i)
      print >>log, "point charge file:", i
      #write mm point charge file
      write_mm_charge_file(fragment_extracts=fq_ext,
                                      index=i)



if (__name__ == "__main__"):
  t0 = time.time()
  args = sys.argv[1:]
  del sys.argv[1:]
  run(args[0], log)
  print >> log, "Time: %6.4f" % (time.time() - t0)
