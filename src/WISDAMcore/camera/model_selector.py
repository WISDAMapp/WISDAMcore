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


from __future__ import annotations
from WISDAMcore.camera.base_perspective import CameraBasePerspective, CameraType
from WISDAMcore.camera.opencv_perspective import CameraOpenCVPerspective
from WISDAMcore.metadata.camera_estimator_metadata import estimate_camera
from WISDAMcore.metadata.camera_alternative_tags import FindCameraCalibration


def select_camera_from_dict(camera_dict: dict) -> CameraBasePerspective:
    """Select camera model from dictionary as received by .param_dict of each camera class

    :param camera_dict: Dictionary as received by camera.param_dict
    :return: Camera Class
    :raises CameraNoImplemented: Camera is not implemented"""

    if camera_dict.get('type') == CameraType.OpenCV.fullname:
        return CameraOpenCVPerspective.from_dict(camera_dict)

    raise NotImplementedError


def estimate_camera_from_meta_dict(meta_dict: dict) -> tuple[CameraBasePerspective | None, int | None, int | None]:
    """Retrieve Camera class by images metadata (Exif, XMP)
    Here you can implement other camera estimations from parameters if in meta dict of xmp or exif present

    :param meta_dict: Meta Dictionary containing exif and xmp tags
    :return: Tuple[CameraModel or None , width of image or None, height of image or None]
             If width, height failed we assume that image can not be used anyhow"""

    res = estimate_camera(meta_dict)

    if res is None:
        return None, None, None

    width, height, focal_pixel, c_x, c_y = res

    camera = None
    if focal_pixel is not None:
        camera = CameraOpenCVPerspective(width=width, height=height, fx=focal_pixel, fy=focal_pixel, cx=c_x, cy=c_y)

    # try to find better camera model in metadata (EXIF/XMP)
    cam_result = FindCameraCalibration.run_all_methods(meta_dict, width=width, height=height)

    if cam_result is not None:
        camera = cam_result

    return camera, width, height
