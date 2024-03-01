# check for installed packages and if they are missing, install them now
# after the initial installation, dependency checks will be disabled
# in setup_state.txt set setup_completed to "False" to trigger a system check / reinstall when enabling omni_trax
from .utils.setup import check_packages
from .utils.setup.CUDA_checks import check_CUDA_installation

from .utils import compute
from .utils import track
from .utils import pose

bl_info = {
    "name": "omni_trax",
    "author": "Fabian Plum",
    "description": "Deep learning-based multi animal tracker",
    "blender": (2, 92, 0),
    "version": (0, 3, 1),
    "location": "",
    "author": "Fabian Plum",
    "warning": "RUN IN ADMINISTRATOR MODE DURING INSTALLATION! "
               "Additional dependencies may conflict with other custom Add-ons.",
    "category": "motion capture"
}


# (un)register module #
def register():
    utils.compute.register()
    utils.track.register()
    utils.pose.register()


def unregister():
    utils.compute.unregister()
    utils.track.unregister()
    utils.pose.unregister()
