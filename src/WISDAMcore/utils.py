# -----------------------------------------------------------------------
# Copyright 2024 Martin Wieser
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------


from typing import Annotated, Literal
import numpy.typing as npt
import numpy as np


Vector2D = Annotated[npt.NDArray[float], Literal[2, ]]
Vector3D = Annotated[npt.NDArray[float], Literal[3, ]]

MaskN_ = Annotated[npt.NDArray[bool], Literal["N", ]]
ArrayN_ = Annotated[npt.NDArray[float], Literal["N", ]]
Array3x3 = Annotated[npt.NDArray[float], Literal[3, 3]]
ArrayNx3 = Annotated[npt.NDArray[float], Literal["N", 3]]
ArrayNx2 = Annotated[npt.NDArray[float], Literal["N", 2]]
ArrayNxN = Annotated[npt.NDArray[float], Literal["N", 2]]


def to_array_nx2(array_like_nx2: list | np.ndarray) -> np.ndarray:

    array_like_nx2 = np.array(array_like_nx2)
    if array_like_nx2.ndim == 1:
        array_like_nx2 = np.array([array_like_nx2])

    if array_like_nx2.shape[0] > 2:
        array_like_nx2 = array_like_nx2[:, :3]

    return array_like_nx2


def to_array_nx3(array_like_nx3: list | np.ndarray) -> np.ndarray:
    array_like_nx3 = np.array(array_like_nx3)
    if array_like_nx3.ndim == 1:
        array_like_nx3 = np.array([array_like_nx3])

    if array_like_nx3.shape[0] > 3:
        array_like_nx3 = array_like_nx3[:, :4]

    return array_like_nx3
