"""TEM Design & Simulation; the stable Python import namespace is temsim."""

__version__ = "0.1.0"
__license__ = "GPL-3.0-only"
__copyright__ = "Copyright (c) 2026 Xuzhao Liu"

# Imported before GUI/NumPy/Numba modules by both application and CLI entries.
from temsim.cpu_resources import initialize_cpu_resources as _initialize_cpu_resources
_initialize_cpu_resources()
