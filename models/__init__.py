"""
MHF-optimized neural operator models
"""

from .fno_mhf import MHFNO, MHFTFNO
from .uno_mhf import MHFUNO
from .gino_mhf import MHF_GINO, MHFFNOGNO
from .gino_mhf_coda import MHF_GINO_CoDA, MHFFNOGNO_CoDA
from .tfno_mhf_coda import MHF_TFNO_CoDA, MHF_TFNO_Baseline
from .tfno_mhf import MHF_TFNO, MHF_TFNO_Progressive, MHF_SpectralConv_TFNO
from .codano_mhf import MHFCODANO

# SFNO - optional (requires torch-harmonics)
try:
    from .sfno_mhf import MHSFNO, MHFSFNO, TORCH_HARMONICS_AVAILABLE
    _MHSFNO_AVAILABLE = True
except (ImportError, NameError):
    MHSFNO = None
    MHFSFNO = None
    TORCH_HARMONICS_AVAILABLE = False
    _MHSFNO_AVAILABLE = False

# RNO - optional
try:
    from .rno_mhf import MHFRNO
    _MHFRNO_AVAILABLE = True
except (ImportError, NameError):
    MHFRNO = None
    _MHFRNO_AVAILABLE = False

# LocalNO - optional
try:
    from .localno_mhf import MHFLocalNO
    _MHFLOCALNO_AVAILABLE = True
except (ImportError, NameError):
    MHFLocalNO = None
    _MHFLOCALNO_AVAILABLE = False

# OTNO - optional
try:
    from .otno_mhf import MHFOTNO
    _MHFOTNO_AVAILABLE = True
except (ImportError, NameError):
    MHFOTNO = None
    _MHFOTNO_AVAILABLE = False

__all__ = [
    "MHFNO",
    "MHFTFNO",
    "MHFUNO",
    "MHSFNO",
    "MHFSFNO",
    "MHF_GINO",
    "MHFFNOGNO",
    "MHF_GINO_CoDA",
    "MHFFNOGNO_CoDA",
    "MHF_TFNO_CoDA",
    "MHF_TFNO_Baseline",
    "MHF_TFNO",
    "MHF_TFNO_Progressive",
    "MHF_SpectralConv_TFNO",
    "MHFCODANO",
    "MHFRNO",
    "MHFLocalNO",
    "MHFOTNO",
    "TORCH_HARMONICS_AVAILABLE",
]
