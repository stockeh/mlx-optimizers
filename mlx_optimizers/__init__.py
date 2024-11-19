from .version import VERSION, VERSION_SHORT

__version__ = VERSION

from .adopt import ADOPT
from .diffgrad import DiffGrad
from .kron import Kron
from .lamb import Lamb
from .madgrad import MADGRAD
from .mars import MARS
from .muon import Muon
from .qhadam import QHAdam
from .shampoo import Shampoo

__all__ = [
    "ADOPT",
    "DiffGrad",
    "Kron",
    "Lamb",
    "MADGRAD",
    "MARS",
    "Muon",
    "QHAdam",
    "Shampoo",
    "__version__",
]
