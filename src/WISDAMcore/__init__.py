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


from WISDAMcore.camera.opencv_perspective import CameraOpenCVPerspective
from WISDAMcore.image.perspective import IMAGEPerspective
from WISDAMcore.mapping.plane import MappingPlane
from WISDAMcore.utils import ArrayNx3, ArrayNx2, Vector2D, Vector3D, Array3x3, ArrayN_, ArrayNxN

from WISDAMcore import cfg


def allow_ballpark_transformations(allow: bool = True):
    cfg._ballpark_transformation = allow


def allow_non_best_transformations(allow: bool = True):
    cfg._only_best_transformation = not allow


