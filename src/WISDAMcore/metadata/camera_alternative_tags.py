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

import inspect

from WISDAMcore.camera.opencv_perspective import CameraOpenCVPerspective


class FindCameraCalibration(object):
    """Class which holds function to estimate camera calibration of different types using metadata
    All methods implemented will be called at once. So if you want to extend different tagging systems
    from different software packages or vendors just add a new methode

    This is only for the camera calibration aka IOR. Exterior orientation aka EOR will be treated in another class"""

    def __init__(self, meta_data: dict, width: float | int, height: float | int):
        self.meta_data = meta_data
        self.width = width
        self.height = height

    def dji_tags(self) -> CameraOpenCVPerspective | None:

        focal_pixel = None
        c_y = self.height / 2.0
        c_x = self.width / 2.0
        if 'XMP:CalibratedFocalLength' in self.meta_data.keys():
            focal_pixel = float(self.meta_data['XMP:CalibratedFocalLength'])

        # For some DjI images calibrated tags have been found
        if {'XMP:CalibratedOpticalCenterX', 'XMP:CalibratedOpticalCenterY'} <= self.meta_data.keys():
            c_x = float(self.meta_data['XMP:CalibratedOpticalCenterX'])
            c_y = float(self.meta_data['XMP:CalibratedOpticalCenterY'])

        # Focal Pixel is not found than do not use that even if Calibrated Center would be found
        if focal_pixel is None:
            return None

        camera = CameraOpenCVPerspective(width=self.width, height=self.height,
                                         fx=focal_pixel, fy=focal_pixel, cx=c_x, cy=c_y)
        return camera

    def pix4d_tags(self):

        return None

    @staticmethod
    def run_all_methods(meta_data, width, height) -> tuple:

        v = FindCameraCalibration(meta_data, width, height)
        attrs = (getattr(v, name) for name in dir(v) if not name.startswith('_') )
        methods = filter(inspect.ismethod, attrs)

        cam_result = None
        for method in methods:
            #print(method)
            try:
                cam_result = method()
                if cam_result is not None:
                    break
            except Exception as e:
                #print(e)
                pass

        return cam_result
