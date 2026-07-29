from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from subprocess import Popen, PIPE

def _get_version():
  qrefine_dir = os.path.dirname(os.path.abspath(__file__))
  try:
    shell = Popen(
      ["git", "describe", "--abbrev=6", "--dirty", "--always", "--tags"],
      cwd=qrefine_dir, stdout=PIPE, stderr=PIPE)
    out, err = shell.communicate()
    if shell.returncode == 0 and out:
      return out.decode().strip()
  except OSError:
    pass
  return "unknown"

__version__ = _get_version()
