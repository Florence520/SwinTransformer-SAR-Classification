__version__ = "1.0.1"
import  os
import sys
this_dir = os.path.dirname(__file__)
dir1  = os.path.join(this_dir, "..")
sys.path.insert(0, dir1)

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
