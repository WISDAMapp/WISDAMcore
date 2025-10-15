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

import pytest
import numpy
from WISDAMcore.camera.opencv_perspective import CameraOpenCVPerspective


@pytest.fixture
def camera_class():
    cam_class = CameraOpenCVPerspective(width=5472, height=3648, fx=3753.0,
                                        fy=3754.0,
                                        cx=2721.2, cy=1813.7,
                                        k1=-0.0082, k2=0.00053,
                                        k3=0.0089,
                                        p1=-0.0015, p2=-0.00086)
    assert int(cam_class.type)
    return cam_class


def test_pixel_image_to_camera_crs(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)

    # The principal point will not be distorted
    result = camera_class.pixel_image_to_camera_crs(prc_pt, img_size)

    assert result[0].tolist() == [0, 0, -1]


def test_pts_camara_crs_to_image_pixel(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)

    # The principal point will not be distorted
    result = camera_class.pts_camara_crs_to_image_pixel(numpy.array([0, 0, 1]), img_size)

    assert result[0].tolist() == prc_pt.tolist()


def test_distorted_to_undistorted(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)
    result = camera_class.distorted_to_undistorted(prc_pt, img_size)
    assert result[0].tolist() == prc_pt.tolist()
    test_points = numpy.array([[100, 100]])
    result = camera_class.distorted_to_undistorted(test_points, img_size)
    result_2 = camera_class.undistorted_to_distorted(result, img_size)
    assert numpy.linalg.norm(test_points - result_2) < 0.1


def test_undistorted_to_distorted(camera_class):
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    prc_pt = camera_class.principal_point(img_size)

    # The principal point will not be distorted
    result = camera_class.distorted_to_undistorted(prc_pt, img_size)

    assert result[0].tolist() == prc_pt.tolist()


def test_image_points_inside(camera_class):
    # Test with image points in camera calibration size
    image_points = numpy.array([[0, 0, ], [100, 100]])
    valid_index = camera_class.undistorted_image_points_inside(image_points, image_size=None)

    assert not bool(valid_index[0])
    assert bool(valid_index[1])

    # Test with image size
    img_size = (camera_class.calibration_width, camera_class.calibration_height)
    valid_index = camera_class.undistorted_image_points_inside(image_points,image_size=img_size)

    assert not bool(valid_index[0])
    assert bool(valid_index[1])
