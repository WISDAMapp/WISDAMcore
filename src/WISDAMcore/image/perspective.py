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
import numpy as np
from pyproj import CRS
from shapely import geometry

from WISDAMcore.utils import ArrayNx3, ArrayNx2, Vector3D, MaskN_

from WISDAMcore.camera.base_perspective import CameraBasePerspective
from WISDAMcore.camera.model_selector import select_camera_from_dict
from WISDAMcore.image.base_class import ImageBase, ImageType
from WISDAMcore.mapping.base_class import MappingBase
from WISDAMcore.transform.coordinates import CoordinatesTransformer
from WISDAMcore.transform.rotation import Rotation

from WISDAMcore.exceptions import MappingError


class IMAGEPerspective(ImageBase):
    """Class for perspective images.
    The image is geo-referenced if position, orientation, camera and crs are specified.
    The camera class used should be implemented so that you can use also resampled images"""

    def __init__(self, width: float | int, height: float | int,
                 camera: CameraBasePerspective | None = None, crs: CRS | None = None,
                 position: Vector3D | None = None, orientation: Rotation | None = None,
                 mapper: MappingBase | None = None):

        super().__init__(mapper=mapper, crs=crs, width=width, height=height)

        self._position: Vector3D | None = position
        self._orientation: Rotation | None = orientation

        self._camera = camera

    @property
    def type(self) -> ImageType:
        return ImageType.Perspective

    @classmethod
    def from_dict(cls, param_dict: dict, mapper: MappingBase | None = None) -> IMAGEPerspective | None:

        width = param_dict['width']
        height = param_dict['height']

        if width == 0 or height == 0:
            raise ValueError("Image width and height can not be 0")

        position = None
        if param_dict.get('position') is not None:
            position = np.array(param_dict.get('position'))

        orientation = None
        if param_dict.get('orientation_matrix') is not None:
            r_matrix = np.array(param_dict.get('orientation_matrix'))
            orientation = Rotation(rotation_matrix=r_matrix)

        crs = None
        if param_dict.get('crs') is not None:
            crs = CRS.from_wkt(param_dict['crs'])

        camera = None
        if param_dict.get('camera') is not None:
            camera = select_camera_from_dict(param_dict['camera'])

        image = cls(width=width, height=height,
                    camera=camera, position=position, orientation=orientation, crs=crs, mapper=mapper)

        return image

    @property
    def param_dict(self) -> dict:
        param_dict = {'type': ImageType.Perspective.fullname,
                      'width': self._width,
                      'height': self._height,
                      'position': self._position.tolist() if self._position is not None else None,
                      'orientation_matrix':
                          self._orientation.matrix.tolist() if self._orientation is not None else None,
                      'camera': self._camera.param_dict if self._camera is not None else None,
                      'crs': self.crs_wkt}

        return param_dict

    @property
    def is_geo_referenced(self) -> bool:
        if self._position is not None \
                and self._orientation is not None \
                and self._crs is not None \
                and self._camera is not None:
            return True
        return False

    @property
    def camera(self):
        return self._camera

    @property
    def center(self) -> tuple:
        """This is not the image center but the principle point"""
        return tuple(self._camera.principal_point(self.shape))

    def image_points_inside(self, point_image_coordinates: ArrayNx2) -> MaskN_:
        """Calculate if image point/points are inside the image borders.
        Within the camera class the border shape of the distortion is generated during initialization.

        :param point_image_coordinates: Array of (nx2) with the image points in image coordinates
        :return: boolean array for points with True if inside image size
        """

        return self.camera.undistorted_image_points_inside(point_image_coordinates, self.shape)

        # old version using discarded parameter outer dist
        # min_border = np.logical_and(point_image_coordinates[:, 0] >= -outer_dist,
        #                            point_image_coordinates[:, 1] >= -outer_dist)
        # max_border = np.logical_and(point_image_coordinates[:, 0] < self.width + outer_dist,
        #                            point_image_coordinates[:, 1] < self.height + outer_dist)
        # valid_index = np.where(np.logical_and(min_border, max_border))[0]
        #
        # if not valid_index.size:
        #    return None
        # return valid_index

    def position_to_crs(self, crs_t: CRS) -> CoordinatesTransformer | None:
        if self._position is not None and self._crs is not None:
            return CoordinatesTransformer.from_crs(self._crs, crs_t, self._position)
        return None

    def project(self,
                points_world_crs: ArrayNx3,
                crs_srs: CRS | None = None, to_distorted: bool = True) -> tuple[ArrayNx2, MaskN_] | None:
        """Calculate projection of 3d points into image. If points was outsize image size and to_distortion is true
        the undistorted projection will be returned

        :param points_world_crs: Array of (nx3) with the point positions
        :param crs_srs: The coordinate system of the input coordinates
        :param to_distorted: Default True, if False only projection to image crs is done without distortion correction
        :return: Points in image space
        """

        if not self.is_geo_referenced:
            return None

        points_world_crs = np.array(points_world_crs)

        # In case of points_world_crs is a single point we make a 2dim np-array from it
        if points_world_crs.ndim < 2:
            points_world_crs = np.array([points_world_crs])

        if crs_srs is None:
            crs_srs = CRS.from_epsg(4979)

        try:
            coo = CoordinatesTransformer.from_crs(crs_srs, self._crs, points_world_crs)
        except:
            return None

        if coo is None:
            return None

        point_3d_crs = coo.coordinates

        # Bring 3d points into Camera CRS by using rotation matrix and projection center
        pts_camera_crs = np.matmul(self._orientation.matrix.T, (point_3d_crs - self._position).T).T

        pts_image_crs = self._camera.pts_camara_crs_to_image_pixel(pts_camera_crs, self.shape, to_distorted=False)

        # Here we check if the undistorted image point is inside the border for which distorted points are inside
        # This is important as non-linear function of distortion,
        # would map extreme far points back to the image due to the higher order polynomials
        #
        valid_pixel = self._camera.undistorted_image_points_inside(pts_image_crs, image_size=None)

        # Only the valid pixel which are inside the distortion border are distorted
        pts_image_crs[valid_pixel, :] = self._camera.pts_camara_crs_to_image_pixel(pts_camera_crs[valid_pixel, :],
                                                                                   self.shape,
                                                                                   to_distorted=True)

        return pts_image_crs, valid_pixel

    def pixel_to_ray_vector(self, pixel_pos: ArrayNx2, is_undistorted: bool = False) -> ArrayNx3 | None:
        """Calculate ray vectors of image points in camera CRS

        :param pixel_pos: Array of (nx3) with the pixel positions
        :param is_undistorted: Flag if undistorted coordinates are given
        :return: Array with the rays vectors with the direction of the ray
        """

        if not self.is_geo_referenced:
            return None

        pixel_pos = np.array(pixel_pos)
        if pixel_pos.ndim < 2:
            pixel_pos = np.array([pixel_pos])

        pts_camera_crs = self._camera.pixel_image_to_camera_crs(pixel_pos, self.shape, is_undistorted)

        line_vec = np.matmul(self._orientation.matrix, pts_camera_crs.T).T
        line_vec = np.divide(line_vec.T, np.linalg.norm(line_vec, axis=1)).T

        return line_vec

    def map_center_point(self, mapper_user: MappingBase | None = None) -> tuple[Vector3D, float] | None:

        if not self.is_geo_referenced:
            raise MappingError("Image is not geo-referenced")

        if self.mapper is None and mapper_user is None:
            raise MappingError("No mapper specified")

        mapper_to_use = self.mapper
        if mapper_user is not None:
            mapper_to_use = mapper_user

        ray_vector = self.pixel_to_ray_vector(self.center)

        try:
            point_center_3d = mapper_to_use.map_coordinates_from_rays(ray_vector, np.array([self._position]), self._crs)
        except MappingError as e:
            raise e

        if point_center_3d is not None:
            dist = np.linalg.norm(self._position - point_center_3d)
            gsd = dist / self._camera.focal_length_for_gsd_in_pixel
            return point_center_3d[0, :], gsd

        return None

    def map_footprint(self, mapper_user: MappingBase | None = None) -> tuple[ArrayNx3, float, float] | None:

        if not self.is_geo_referenced:
            raise MappingError("Image is not geo-referenced")

        if self.mapper is None and mapper_user is None:
            raise MappingError("No mapper specified")

        mapper_to_use = self.mapper
        if mapper_user is not None:
            mapper_to_use = mapper_user

        footprint_points_2d = np.array([[0, 0],
                                        [self.width, 0],
                                        [self.width, self.height],
                                        [0, self.height]])

        ray_vector = self.pixel_to_ray_vector(footprint_points_2d)
        ray_pos = np.ones(ray_vector.shape) * self._position
        try:
            footprint_points_3d = mapper_to_use.map_coordinates_from_rays(ray_vector, ray_pos, self._crs)
        except MappingError as e:
            raise e

        if footprint_points_3d is not None:
            footprint_geom = geometry.Polygon(footprint_points_3d)
            area = float(np.round(footprint_geom.area))
            gsd = float(np.round(np.sqrt(area / (self.width * self.height)), 4))

            return footprint_points_3d, gsd, area

        return None

    def map_points(self,
                   points_image: ArrayNx2,
                   mapper_user: MappingBase | None = None) -> tuple[ArrayNx3, float]:

        if not self.is_geo_referenced:
            raise MappingError("Image is not geo-referenced")

        if self.mapper is None and mapper_user is None:
            raise MappingError("No mapper specified")

        mapper_to_use = self.mapper
        if mapper_user is not None:
            mapper_to_use = mapper_user

        # build ray vectors of image points
        ray_vector = self.pixel_to_ray_vector(points_image)
        # build ray start points for all rays
        ray_pos = np.ones(ray_vector.shape) * self._position

        # Map points using either the images specified mapper or the user mapper provided
        try:
            points_3d = mapper_to_use.map_coordinates_from_rays(ray_vector, ray_pos, self._crs)
        except MappingError as e:
            raise e

        dist = np.mean(np.linalg.norm(self._position - points_3d, axis=1))
        gsd = dist / self._camera.focal_length_for_gsd_in_pixel

        return points_3d, gsd
