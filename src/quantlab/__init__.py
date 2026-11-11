"""NBA Quant Lab: player box-score forecasting and possession-level game simulation."""

import sys

if sys.platform == "win32":
    # numba and LightGBM load the Microsoft C++ runtime (msvcp140.dll, vcomp140.dll) from the system, which a PC
    # without the Visual C++ Redistributable lacks; scikit-learn's Windows wheels ship and preload both, so importing
    # it first lets those imports succeed anyway.
    try:
        import sklearn  # noqa: F401
    except ImportError:  # pragma: no cover
        pass
