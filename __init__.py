from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from subprocess import Popen, PIPE

_version_cache = None

def _get_version():
  global _version_cache
  if _version_cache is not None:
    return _version_cache
  qrefine_dir = os.path.dirname(os.path.abspath(__file__))
  try:
    shell = Popen(
      ["git", "describe", "--abbrev=6", "--dirty", "--always", "--tags"],
      cwd=qrefine_dir, stdout=PIPE, stderr=PIPE)
    out, err = shell.communicate()
    if shell.returncode == 0 and out:
      _version_cache = out.decode().strip()
      return _version_cache
  except OSError:
    pass
  _version_cache = "unknown"
  return _version_cache

def __getattr__(name):
  if name == "__version__":
    return _get_version()
  raise AttributeError("module %r has no attribute %r" % (__name__, name))
