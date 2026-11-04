"""Python ERC cam design utilities.

The package ports the support-function cam profile workflow from
``ERC_model.mlx`` into torch-native classes that can be called from an
Isaac Lab codesign loop.
"""

from .designer import ERCDesignConfig, ERCDesignResult, ERCDesigner, interpolate_output_torque_table
from .profiles import PiecewiseLinearTorqueProfile, ProfileParameterization
from .springs import Spring, SpringCatalog

__all__ = [
    "ERCDesignConfig",
    "ERCDesignResult",
    "ERCDesigner",
    "interpolate_output_torque_table",
    "PiecewiseLinearTorqueProfile",
    "ProfileParameterization",
    "Spring",
    "SpringCatalog",
]
