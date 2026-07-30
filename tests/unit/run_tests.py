from __future__ import division
from __future__ import print_function
import os
import time
import shutil
import iotbx.pdb
import libtbx.load_env
from libtbx import easy_run
from scitbx.array_family import flex
from libtbx import easy_mp
import traceback
from libtbx.utils import Sorry

qrefine = libtbx.env.find_in_repositories("qrefine")
qr_unit_tests = os.path.join(qrefine, "tests","unit")
qr_unit_tests_data = os.path.join(qr_unit_tests,"data_files")

def setup_helix_example(pdb_name = "m00_good.pdb",
                        mtz_name = "m00_good.mtz"):
  # Read good model and compute data from it.
  xrs_good = iotbx.pdb.input(
    file_name = os.path.join(qr_unit_tests_data,
    pdb_name )).xray_structure_simple()
  f_obs = abs(xrs_good.structure_factors(d_min=1.5).f_calc())
  r_free_flags_data = flex.bool()
  for i in range(f_obs.data().size()):
    if(i%10 == 0): r_free_flags_data.append(True)
    else:          r_free_flags_data.append(False)
  r_free_flags = f_obs.array(data = r_free_flags_data)
  mtz_dataset = f_obs.as_mtz_dataset(column_root_label="F-obs")
  mtz_dataset.add_miller_array(
    miller_array      = r_free_flags,
    column_root_label = "R-free-flags")
  mtz_object = mtz_dataset.mtz_object()
  mtz_object.write(file_name = mtz_name)
  xrs_poor = iotbx.pdb.input(
    file_name = os.path.join(qr_unit_tests_data,
      "m00_poor.pdb")).xray_structure_simple()
  return xrs_good,xrs_poor,f_obs,r_free_flags

def run_cmd(prefix,args,pdb_name = "m00_poor.pdb",
            mtz_name = "m00_good.mtz"):
  test_folder_name = prefix
  cmd = [
    "qr.refine",
    mtz_name,
    os.path.join(qr_unit_tests_data,pdb_name)]
  for arg in args:
    cmd.append(arg)
  cmd.append("output_folder_name=%s"%test_folder_name)
  cmd.append("> %s.log"%prefix)
  if(1): print(" ".join(cmd))
  rc = easy_run.go(" ".join(cmd))
  if rc.return_code != 0:
    rc.show_stderr()
    rc.show_stdout()
    raise SystemExit(f"A command within a test failed!")
  return rc.return_code

def runner(function, prefix, disable=False):
  prefix = "qrefine_"+prefix
  import sys
  rc = 0
  try:
    if(disable):
      print(prefix + ": Skipped (not recommended, do something about it!)")
    else:
      t0 = time.time()
      os.mkdir(prefix)
      os.chdir(prefix)
      function(prefix = prefix)
      print(prefix + ":  OK  " + "Time: %6.2f (s)" % (time.time() - t0))
  except Exception as e:
      print(prefix, "FAILED", str(e))
      print(traceback.format_exc())
      rc=1
  os.chdir('..')
  assert not rc, "%s rc: %s" % (prefix, rc)
  return rc

def run(nproc=1,
        only_i=None,
        non_mopac_only=False):
  cwd = os.getcwd()
  assert cwd.find(' ')==-1, 'test do not work in directory with a space " "'
  t0=time.time()
  print('Running tests on %d processors' % nproc)
  # Individual test runner
  # NOTE: this runs inside an easy_mp worker subprocess when nproc>1. Its
  # stdout is NOT captured/serialized by easy_mp (only stderr is), so it
  # must not print() anything itself -- concurrent workers would scramble
  # the terminal. All output goes to a per-test log file instead; only the
  # parent (the loop in run(), below) is safe to print from.
  def _run_test(file_name, in_separate_directory=True):
    t00=time.time()
    fn = file_name.split('.')[0]
    if in_separate_directory:
      if os.path.exists(fn):
        return {"return_code": 1,
                "elapsed": time.time()-t00,
                "log_path": None,
                "message": f"stale directory '{fn}' already exists from a "
                  "previous run -- remove it before rerunning "}
      os.mkdir(fn)
      os.chdir(fn)
    full_test_file_name = os.path.join(qr_unit_tests,file_name)
    log_path = os.path.abspath("%s.log"%fn)
    rc = easy_run.go("qrefine.python %s"%(full_test_file_name))
    with open(log_path, "w") as log:
      log.write("Running test: %s in folder: %s\n"%(full_test_file_name,fn))
      for line in rc.stdout_lines:
        log.write(line+"\n")
      for line in rc.stderr_lines:
        log.write(line+"\n")
      log.write('Time (this test): %6.2f (s)\n' % (time.time()-t00))
    if in_separate_directory:
      os.chdir('..')
    return {"return_code": rc.return_code,
            "elapsed": time.time()-t00,
            "log_path": log_path}
  # Collect test files
  tests = []
  for fn in os.listdir(qr_unit_tests):
    if(fn.startswith("tst_") and fn.endswith(".py")):
      i_test = fn[:].replace("tst_","").replace(".py","")
      i_test = i_test[0].replace('0','')+i_test[1]
      if(only_i is not None):
        if(only_i == i_test):
          tests.append(fn)
      else:
        tests.append(fn)
  tests.sort()
  if non_mopac_only:
    remove = []
    for i, file_name in enumerate(tests):
      f=open(os.path.join(qr_unit_tests, file_name), 'r')
      lines=f.read()
      del f
      if lines.lower().find('mopac')>-1:
        remove.append(i)
    if remove:
      remove.reverse()
      for r in remove:
        print('Removing test %s from list' % tests[r])
        del tests[r]
  print("Following tests will be executed:")
  print(" ".join(tests))
  #
  n_total = len(tests)
  results = []
  in_separate_directory=True # not(nproc==1)
  for i, file_name in enumerate(tests):
    tests[i]=tuple([file_name, in_separate_directory])
  #
  for i, (args, res, err_str) in enumerate(easy_mp.multi_core_run(
                                                    _run_test,
                                                    tests,
                                                    nproc,
                                                    )):
    file_name = args[0]
    fn = file_name.split('.')[0]
    message = None
    if isinstance(res, dict):
      rc = res["return_code"]
      elapsed = res["elapsed"]
      log_path = res["log_path"]
      message = res.get("message")
    else:
      # worker crashed before returning (e.g. exception outside easy_run.go)
      rc = 1
      elapsed = None
      log_path = None
    status = "OK" if rc == 0 else "FAILED"
    elapsed_str = ("%6.2f (s)" % elapsed) if elapsed is not None else "   n/a"
    print('[%d/%d] %-12s %-6s %s' % (i+1, n_total, fn, status, elapsed_str))
    if message:
      print('    %s' % message)
    if err_str:
      # worker-process stderr (crash traceback etc.), not the test's own
      # captured output -- route to the per-test log when one exists.
      if log_path:
        with open(log_path, "a") as log:
          log.write("--- worker stderr ---\n")
          log.write(err_str)
      else:
        print('Error output from %s' % fn)
        print(err_str)
    results.append({"name": fn, "status": status, "elapsed": elapsed,
                     "log_path": log_path, "message": message})
  failed_results = [r for r in results if r["status"] == "FAILED"]
  print('='*60)
  print('Test summary: %d run, %d passed, %d failed  (total time: %6.2f (s))' % (
    n_total, n_total-len(failed_results), len(failed_results), time.time()-t0))
  if failed_results:
    print('FAILED tests:')
    for r in failed_results:
      elapsed_str = ("%6.2f (s)" % r["elapsed"]) if r["elapsed"] is not None else "   n/a"
      if r["message"]:
        print('  %-12s %s   %s' % (r["name"], elapsed_str, r["message"]))
      else:
        print('  %-12s %s   log: %s' % (r["name"], elapsed_str, r["log_path"]))
  print('='*60)
  if failed_results:
    return 1
  return 0

if(__name__ == "__main__"):
  t0 = time.time()
  args = sys.argv[1:]
  nproc=1
  if args: nproc=int(args[0])
  rc = run(nproc=nproc)
  print("Total time (all tests): %6.2f"%(time.time()-t0))
  if rc:
    assert not rc
  else:
    print("OK")
