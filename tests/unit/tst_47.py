import os

from libtbx import easy_run

from qrefine.tests.unit import run_tests


def read_log(prefix):
  with open("%s.log" % prefix) as f:
    return f.read()

def run_expect_ok(prefix, args, expected_message, unexpected_message=None):
  run_tests.run_cmd(prefix, args=args)
  log = read_log(prefix)
  assert expected_message in log, \
    f"Expected message not found in log:\n{expected_message}\n---\n{log}"
  if(unexpected_message is not None):
    assert unexpected_message not in log, \
      f"Unexpected message found in log:\n{unexpected_message}\n---\n{log}"

def run_expect_sorry(prefix, args, expected_message):
  cmd = ["qr.refine", "m00_good.mtz",
    os.path.join(run_tests.qr_unit_tests_data, "m00_poor.pdb")]
  cmd.extend(args)
  cmd.append(f"output_folder_name={prefix}")
  rc = easy_run.go(" ".join(cmd))
  assert rc.return_code != 0, \
    "Expected failure (missing data_weight) but command succeeded."
  out = "\n".join(rc.stdout_lines)
  assert expected_message in out, \
    f"Expected error message not found in output:\n{out}"

def run(prefix):
  """
  Exercise all combinations of refine.skip_weight_search and
  refine.data_weight, and confirm via the qr.refine log which code path
  actually ran (not just that the process exited cleanly):
    skip_weight_search=True,  data_weight given     -> succeeds, weight search skipped, user weight used as-is
    skip_weight_search=True,  data_weight not given  -> raises Sorry
    skip_weight_search=False, data_weight given      -> succeeds, weight search seeded from user value
    skip_weight_search=False, data_weight not given   -> succeeds, weight search computes a default
  """
  run_tests.setup_helix_example()

  common = ["restraints=cctbx", "clustering=False", "minimizer=lbfgsb",
    "number_of_micro_cycles=1", "max_iterations_refine=5"]

  # skip_weight_search=True, data_weight given: weight search must be
  # skipped and the user-supplied weight used as-is.
  run_expect_ok(prefix,
    args = common + ["skip_weight_search=True", "data_weight=0"],
    expected_message =
      "Using user-supplied data weight (weight search skipped): 0.0",
    unexpected_message = "Optimal weight search:")

  # skip_weight_search=True, data_weight not given: should raise Sorry.
  run_expect_sorry(prefix,
    args = common + ["skip_weight_search=True"],
    expected_message =
      "refine.data_weight must be specified when refine.skip_weight_search=True.")

  # skip_weight_search=False (default), data_weight given: weight search
  # must run, seeded from the user-supplied value.
  run_expect_ok(prefix,
    args = common + ["data_weight=5"],
    expected_message =
      "Starting weight search from user-supplied data weight: 5.0")

  # skip_weight_search=False (default), data_weight not given: weight
  # search must run and compute its own starting value (no user-supplied
  # value to report).
  run_expect_ok(prefix,
    args = common,
    expected_message = "Optimal weight search:",
    unexpected_message = "Starting weight search from user-supplied data weight")

if(__name__ == "__main__"):
  prefix = os.path.basename(__file__).replace(".py","")
  run_tests.runner(function=run, prefix=prefix, disable=False)
