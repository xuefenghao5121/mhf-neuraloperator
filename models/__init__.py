"""
MHF-optimized neural operator models
"""

from .fno_mhf import MHFNO, MHFTFNO
from .uno_mhf import MHFUNO
# from .sfno_mhf import MHSFNO  # Requires torch_harmonics
from .gino_mhf import MHF_GINO, MHFFNOGNO
from .gino_mhf_coda import MHF_GINO_CoDA, MHFFNOGNO_CoDA
from .tfno_mhf_coda import MHF_TFNO_CoDA, MHF_TFNO_Baseline
from .codano_mhf import MHFCODANO
from .rno_mhf import MHFRNO
# from .localno_mhf import MHFLocalNO  # Requires torch_harmonics
from .otno_mhf import MHFOTNO

__all__ = [
    "MHFNO",
    "MHFTFNO",
    "MHFUNO",
    # "MHSFNO",
    "MHF_GINO",
    "MHFFNOGNO",
    "MHF_GINO_CoDA",
    "MHFFNOGNO_CoDA",
    "MHF_TFNO_CoDA",
    "MHF_TFNO_Baseline",
    "MHFCODANO",
    "MHFRNO",
    # "MHFLocalNO",
    "MHFOTNO",
]
