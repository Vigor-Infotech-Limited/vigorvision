# vigorvision  AGPL-3.0 License - https://vigorvision.com/license

__version__ = "8.3.140"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from vigorvision.models import NAS, RTDETR, SAM, Vision, VisionE, FastSAM, VisionWorld
from vigorvision.utils import ASSETS, SETTINGS
from vigorvision.utils.checks import check_Vision as checks
from vigorvision.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "Vision",
    "VisionWorld",
    "VisionE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
