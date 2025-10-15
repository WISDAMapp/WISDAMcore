# -*- coding: utf-8 -*-
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
from abc import abstractmethod
import numpy as np
from enum import Enum

from shapely import geometry

from WISDAMcore.utils import ArrayNx3, ArrayNx2, Vector2D, MaskN_


class CameraType(Enum):
    """Enum with the camera types.
    Add new camera types here as well"""

    OpenCV = 1, 'OpenCV'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class PostMetaCaller(type):
    def __call__(cls, *args, **kwargs):
        # here is "before __new__ is called"
        instance = type.__call__(cls, *args, **kwargs)
        instance.__post_init__()

        return instance


class CameraBasePerspective(metaclass=PostMetaCaller):
    """ Base class for perspective cameras which is used as transition from the camera crs to the image pixel crs
    This class does not hold any geo-reference information. Transition from and to world CRS is done by the image class.

    CRS System of image pixel coordinate system: X is pointing right and y is pointing down from top left pixel edge
    CRS System of camera coordinate system: X is pointing right, y pointing up and z pointing backwards.

    This class deals with functions needed to get from the image to the camera crs.
    Origin shifts and image calibration is used to deal with resampled images and different camera models

    The basic functions within that Base Class are used to call the private functions which will need to be implemented.
    You will need to reimplement these functions for other camera models:
    class method - "from_dict" \n
    property - "type" should be according to the CameraType you added \n
    property - "param_dict"\n
    property - "focal_length_for_gsd_in_pixel"\n
    property - "_origin"\n
    property - "_principal_point"
    methode - "_to_distorted_points"\n
    methode - "_to_undistorted_points",\n
    methode - "_pixel_undistorted_to_vector_in_camera_crs"\n
    methode - "_pts_camara_crs_to_pixel_undistorted"
    """

    def __init__(self, width: int, height: int):

        self.calibration_width = width
        self.calibration_height = height

        # initial border
        self.distortion_border = geometry.Polygon([[0, 0],
                                                   [self.calibration_width, 0],
                                                   [self.calibration_width, self.calibration_height],
                                                   [0, self.calibration_height]])

    def __post_init__(self):
        distortion_border = self._generate_distortion_border()
        self.distortion_border = geometry.Polygon(distortion_border)

    @classmethod
    @abstractmethod
    def from_dict(cls, para_dict: dict):
        """Initialize class based on a dict which is retrieved from param_dict"""
        # Has to be implemented for each camera Class
        pass

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        """get camera parameters as dict according to the implementation of from_dict"""
        # Has to be implemented for each camera Class
        pass

    @property
    @abstractmethod
    def type(self) -> CameraType:
        """Return the type of the camera as CameraType"""
        pass

    @property
    @abstractmethod
    def focal_length_for_gsd_in_pixel(self):
        pass

    @property
    @abstractmethod
    def _origin(self):
        """One has to implement the pixel origin of the camera model as offset to the used
        pixel system. Our system is using the upper left corner as 0,0
        For example OpenCVs pixel coordinate system is defined with left upper CENTER is (0,0) therefore
        we need to apply an offset and return np.array([0.5,0.5])"""
        pass

    @property
    @abstractmethod
    def _principal_point(self) -> Vector2D:
        """principal point in camera size

        :return: np.ndARRAY(2,) with the principal point in camera calibration size
        """
        # Has to be implemented for each camera Class
        pass

    def _generate_distortion_border(self, points_between: int = 5) -> ArrayNx2:

        img_points = np.vstack((
                                np.linspace([0, 0], [self.calibration_width, 0], points_between+1, axis=0,
                                            endpoint=False),
                                np.linspace([self.calibration_width, 0],
                                            [self.calibration_width, self.calibration_height],
                                            points_between+1, axis=0, endpoint=False),
                                np.linspace([self.calibration_width, self.calibration_height],
                                            [0, self.calibration_height],
                                            points_between + 1, axis=0, endpoint=False),
                                np.linspace([0, self.calibration_height],
                                            [0, 0],
                                            points_between + 1, axis=0, endpoint=False)))

        distortion_border = self._to_undistorted_points(img_points)
        return distortion_border

    @abstractmethod
    def _to_distorted_points(self, pts_undistorted_pixel: ArrayNx2) -> np.ndarray:
        """function for calculating distorted points from undistorted ones in original calibration image size

        :param pts_undistorted_pixel: Array of (nx2) with the image coordinates
        :return: Array with distorted points
        """
        # Has to be implemented for each camera Class
        pass

    @abstractmethod
    def _to_undistorted_points(self, pts_distorted_pixel: ArrayNx2) -> np.ndarray:
        """function for calculating undistorted points from distorted ones

        :param pts_distorted_pixel: Array of (nx2) with the image coordinates
        :return: Array with undistorted points
        """
        # Has to be implemented for each camera Class
        pass

    @abstractmethod
    def _image_crs_to_vector_in_camera_crs(self, points_image: ArrayNx2) -> np.ndarray:
        """function for calculating vector in camera crs of undistorted image points

        :param points_image: Array of (nx2) with the image coordinates
        :return: Array with vectors of the line of sight of image positions
        """
        # Has to be implemented for each camera Class
        pass

    @abstractmethod
    def _camara_crs_to_image_crs(self, pts_undistorted_pixel: ArrayNx3) -> np.ndarray:
        """function for calculating undistorted points from points in camera crs

        :param pts_undistorted_pixel: Array of (nx3) with the image coordinates
        :return: Array with undistorted points
        """
        # Has to be implemented for each camera Class
        pass

    @property
    def origin(self):
        return self._origin

    def principal_point(self, image_size: tuple) -> Vector2D:
        return self._pixel_calibration_to_image_size(self._principal_point, image_size)[0]

    def _pixel_image_to_calibration_size(self, pixel_from_image: ArrayNx2, image_size: Vector2D) -> np.ndarray:
        """Calculate scaled image coordinates to calibration size

        :param pixel_from_image: Array of (nx2) with the image coordinates
        :param image_size: Size of the current image - taken from file
        :return: Array with the image positions scaled to calibration image size
        """

        if pixel_from_image.ndim < 2:
            pixel_from_image = np.array([pixel_from_image])

        px_normalized = (pixel_from_image / image_size)
        px_calibration_size = px_normalized * np.array([self.calibration_width,
                                                        self.calibration_height])
        px_calibration_size -= self._origin
        return px_calibration_size

    def _pixel_calibration_to_image_size(self, pixel_from_calib_size: ArrayNx2, image_size: tuple):
        """Calculate scaled image coordinates to image file size

        :param pixel_from_calib_size: Array of (nx2) with the image coordinates
        :param image_size: Size of the current image - taken from file
        :return: Array with the image positions scaled to file image size
        """
        image_size = np.array(image_size)
        if pixel_from_calib_size.ndim < 2:
            pixel_from_calib_size = np.array([pixel_from_calib_size])

        img_calib_size = np.array([self.calibration_width, self.calibration_height])
        px_normalized = (pixel_from_calib_size / img_calib_size)
        px_image_size = px_normalized * image_size

        # Correct px image size for origin shift from the camera model
        px_image_size += self.origin
        return px_image_size

    def undistorted_image_points_inside(self, points_image_crs: ArrayNx2,
                                        image_size: tuple | None = None) -> MaskN_:
        """Returns a bool array of size Nx1 with True if image point is inside the image border.
        This as takes into account the distortion border of the camera model"""

        pixel_calibration_size = points_image_crs * 1.0
        if image_size is not None:
            pixel_calibration_size = self._pixel_image_to_calibration_size(points_image_crs, image_size=image_size)

        valid_mask = np.full((pixel_calibration_size.shape[0]), False)
        for idx, pt in enumerate(pixel_calibration_size):
            valid_mask[idx] = self.distortion_border.contains(geometry.Point(pt))

        return valid_mask

    def pixel_image_to_camera_crs(self, points_image_crs: ArrayNx2, image_size: tuple,
                                  is_undistorted: bool = False) -> np.ndarray:
        """Calculating vector in camera crs of image points.
        Image points are internally scaled to calibration image size for validity of distortion parameters

        :param points_image_crs: Array of (nx2) with the image coordinates
        :param image_size: NPArray (width,height) of file size
        :param is_undistorted: Flag boolean if undistorted image coordinates are provided
        :return: Array with vectors of the line of sight of image positions
        """
        points_image_crs = np.array(points_image_crs)
        if points_image_crs.ndim < 2:
            points_image_crs = np.array([points_image_crs])

        points_image_crs = points_image_crs

        # Points will be undistorted.
        if not is_undistorted:
            points_image_crs = self.distorted_to_undistorted(points_image_crs, image_size)

        # Points will be scaled to calibration size as pixel size is defined for this
        points_image_scaled_to_calibration = self._pixel_image_to_calibration_size(points_image_crs, image_size)
        pts_camera_crs = self._image_crs_to_vector_in_camera_crs(points_image_scaled_to_calibration)
        return pts_camera_crs

    def pts_camara_crs_to_image_pixel(self, points_camera_crs: ArrayNx3, image_size: tuple,
                                      to_distorted: bool = False) -> np.ndarray:
        """Calculating image coordinates of points in camera crs.
        Image points are internally scaled to calibration image size for validity of distortion parameters

        :param points_camera_crs: Array of (nx3) with the camera crs coordinates
        :param image_size: NPArray (width,height) of file size
        :param to_distorted: Flag boolean if distorted image coordinates should be calculated
        :return: Array with image positions
        """
        points_camera_crs = np.array(points_camera_crs)
        if points_camera_crs.ndim < 2:
            points_camera_crs = np.array([points_camera_crs])

        points_image_calibration_size = self._camara_crs_to_image_crs(points_camera_crs)

        if to_distorted:
            points_image_calibration_size = self._to_distorted_points(points_image_calibration_size)

        points_image_crs_scaled_to_image = self._pixel_calibration_to_image_size(points_image_calibration_size,
                                                                                 image_size)

        points_image_crs_scaled_to_image = points_image_crs_scaled_to_image
        return points_image_crs_scaled_to_image

    def distorted_to_undistorted(self, points_distorted: ArrayNx2, image_size: tuple) -> np.ndarray:
        """Calculating undistorted image coordinates from distorted ones
        Image points are internally scaled to calibration image size for validity of distortion parameters

        :param points_distorted: Array of (nx2) with the image coordinates
        :param image_size: NPArray (width,height) of file size
        :return: Array with undistorted image coordinates
        """
        scaled_pixel = self._pixel_image_to_calibration_size(points_distorted, image_size)
        undistorted_scaled = self._to_undistorted_points(scaled_pixel)
        undistorted_back_scaled_pixel = self._pixel_calibration_to_image_size(undistorted_scaled, image_size)
        return undistorted_back_scaled_pixel

    def undistorted_to_distorted(self, pts_undistorted: ArrayNx3, image_size: tuple) -> np.ndarray:
        """Calculating distorted image coordinates from undistorted ones
        Image points are internally scaled to calibration image size for validity of distortion parameters

        :param pts_undistorted: Array of (nx2) with the image coordinates
        :param image_size: NPArray (width,height) of file size
        :return: Array with distorted image coordinates
        """
        scaled_pixel = self._pixel_image_to_calibration_size(pts_undistorted, image_size)
        distorted_scaled = self._to_distorted_points(scaled_pixel)
        distorted_back_scaled_pixel = self._pixel_calibration_to_image_size(distorted_scaled, image_size)

        return distorted_back_scaled_pixel
