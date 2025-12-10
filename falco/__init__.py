from .config import init_from_mat as ifm  # ← CHANGED: Use relative import
from . import config
from . import model
from . import proper
from . import wfsc

from .setup import *
from .util import *
from .imaging import *
from .dm import *
from .diff_dm import *
from .est import *
from .ctrl import *
from .hlc import *
from .wfsc  import *
from .mask import *
from .plot import *
from .prop import *
from .hexsegmirror import *
from .thinfilm import *
from .zern import *
from ._globals import INFLUENCE_XINETICS
from ._globals import INFLUENCE_BMC_KILO
from ._globals import INFLUENCE_BMC_2K
from ._globals import INFLUENCE_BMC_2K_RES20
