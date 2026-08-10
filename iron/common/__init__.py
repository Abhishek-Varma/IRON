# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities and base classes for IRON operators."""

# IRON dispatches exclusively through the HRX (amdxdna / libhrx) host runtime.
# The backend is selected by the ``NPU_RUNTIME`` env var read inside the ``aie``
# package at import time, so it must be set *before* ``.base`` triggers
# ``import aie`` below. ``setdefault`` lets an explicit override through (e.g.
# for debugging), but the default IRON runtime is HRX.
import os as _os

_os.environ.setdefault("NPU_RUNTIME", "hrx")

from .base import (
    AIEOperatorBase,
    MLIROperator,
    CompositeOperator,
    AIERuntimeArgSpec,
)
from .operator_bases import ChanneledUnaryOperator, BinaryElementwiseOperator
from .context import AIEContext
from .compilation import (
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
