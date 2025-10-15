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
from pathlib import Path
import numpy as np

import rasterio
from pyproj import CRS
from shapely import geometry

from WISDAMcore.utils import ArrayNx3, ArrayNx2, Vector3D, MaskN_, to_array_nx2, to_array_nx3
from WISDAMcore.transform.coordinates import CoordinatesTransformer
from WISDAMcore.image.base_class import ImageBase, ImageType
from WISDAMcore.mapping.base_class import MappingBase
from WISDAMcore.exceptions import CoordinateTransformationError
from WISDAMcore.exceptions import MappingError


class IMAGEOrtho(ImageBase):
    """Image class dealing with orthophotos using rasterio package"""

    def __init__(self, width: float | int, height: float | int,
                 geo_transform: rasterio.Affine | None = None, resolution: float = 0.0, crs: CRS | None = None,
                 mapper: MappingBase | None = None):
        super().__init__(mapper=mapper, crs=crs, width=width, height=height)

        # geo transform
        self.geo_transform = geo_transform

        self._res = resolution

        if self._res == 0.0:
            self._res = float(np.mean(self.geo_transform._scaling))

    @classmethod
    def from_file(cls, path: Path | str, mapper: MappingBase | None = None, crs: CRS | None = None) \
            -> IMAGEOrtho | None:
        """Initialize class directly by a file

        :param path: Path to the file
        :param mapper: Mapper class needed for mapping
        :param crs: CRS which will override the files CRS
        :returns: Image class or None if failed"""
        path = Path(path)

        dataset = rasterio.open(path)
        width = dataset.shape[0]
        height = dataset.shape[1]

        # rasterio returns identity if file has no geo-reference
        # We want that gt is None instead of having an identity matrix
        if dataset.transform.is_identity:
            gt = None
        else:
            gt = dataset.transform

        if crs is None:
            # get crs code from orthophoto and promote to 3d.
            # Actually that would not be needed for the orthophoto
            # but to be sure all pyproj transformations are working
            if dataset.crs is not None:
                crs: CRS = CRS(dataset.crs).to_3d()

        # TODO this should be optimized
        if dataset.crs is not None:
            try:
                linear_unit_factor = dataset.crs.linear_units_factor
                res = float(np.mean(dataset.res) * linear_unit_factor[1])
            except rasterio.errors.CRSError:
                linear_unit_factor = 1.0
                res = float(np.mean(dataset.res) * linear_unit_factor)

        else:
            res = float(np.mean(dataset.res))

        # initialize the image class
        image = cls(width, height, geo_transform=gt, resolution=res, crs=crs, mapper=mapper)

        return image

    @property
    def type(self):
        return ImageType.Orthophoto

    @classmethod
    def from_dict(cls, param_dict: dict, mapper: MappingBase | None = None) -> ImageBase | None:

        width = param_dict['width']
        height = param_dict['height']

        # get geo transform
        gt = None
        if param_dict.get('geo_transform', None) is not None:
            gt = rasterio.Affine(**param_dict['geo_transform'])

        resolution = param_dict.get('resolution', 0.0)

        crs = None
        if param_dict.get('crs', False):
            crs = CRS.from_wkt(param_dict['crs'])

        image = cls(width, height, geo_transform=gt, resolution=resolution, crs=crs, mapper=mapper)

        return image

    @property
    def param_dict(self) -> dict:
        param_dict = {"type": self.type.fullname,
                      'width': self._width,
                      'height': self._height,
                      'geo_transform': self.geo_transform._asdict() if self.geo_transform is not None else None,
                      'resolution': self._res,
                      'crs': None}
        if self.crs is not None:
            param_dict['crs'] = self.crs.to_wkt()

        return param_dict

    @property
    def center(self) -> tuple:
        return self.width / 2.0, self.height / 2.0

    @property
    def is_geo_referenced(self) -> bool:
        if self.geo_transform is not None and self._crs is not None:
            return True
        return False

    def image_points_inside(self, point_image_coordinates: ArrayNx2) -> MaskN_:

        # Format points to be numpy array Nx2
        point_image_coordinates = to_array_nx2(point_image_coordinates)

        min_border = np.logical_and(point_image_coordinates[:, 0] >= 0,
                                    point_image_coordinates[:, 1] >= 0)
        max_border = np.logical_and(point_image_coordinates[:, 0] < (self.width - 1.0),
                                    point_image_coordinates[:, 1] < (self.height - 1.0))
        valid_index = np.logical_and(min_border, max_border)

        return valid_index

    def position_to_crs(self, crs_t: CRS) -> CoordinatesTransformer | None:
        if self.is_geo_referenced:
            pos_wgs84 = self.map_center_point()
            if pos_wgs84 is not None:
                coordinates, gsd = pos_wgs84
                return CoordinatesTransformer.from_crs(crs_s=self.crs, crs_t=crs_t, points=coordinates)
        return None

    def project(self, points_world_crs: ArrayNx3, crs_srs: CRS | None = None) -> tuple[ArrayNx2, MaskN_] | None:

        if not self.is_geo_referenced:
            return None

        if crs_srs is None:
            crs_srs = CRS.from_epsg(4979)

        try:
            coo = CoordinatesTransformer.from_crs(crs_srs, self._crs, points_world_crs)
        except CoordinateTransformationError as e:
            raise CoordinateTransformationError("Error during projection, Probably Coordinate Systems are wrong") from e

        point_3d_crs = coo.coordinates

        x, y = ~self.geo_transform * (point_3d_crs[:, 0], point_3d_crs[:, 1])

        raster_coordinates = np.vstack((x, y)).T
        valid = self.image_points_inside(raster_coordinates)
        return raster_coordinates, valid

    def map_center_point(self, mapper_user: MappingBase | None = None) -> tuple[Vector3D, float] | None:

        if not self.is_geo_referenced:
            raise MappingError("Image is not geo-referenced")

        if self.mapper is None and mapper_user is None:
            raise MappingError("No mapper specified")

        mapper_to_use = self.mapper
        if mapper_user is not None:
            mapper_to_use = mapper_user

        point_center_2d = self.geo_transform * [self.width / 2.0, self.height / 2.0]
        point_center_3d = mapper_to_use.map_heights_from_coordinates(point_center_2d, self._crs)

        if point_center_3d is not None:

            return point_center_3d[0, :], self._res

        return None

    def map_footprint(self, mapper_user: MappingBase | None = None) -> tuple[ArrayNx3, float, float] | None:

        if not self.is_geo_referenced:
            raise MappingError("Image is not geo-referenced")

        if self.mapper is None and mapper_user is None:
            raise MappingError("No mapper specified")

        mapper_to_use = self.mapper
        if mapper_user is not None:
            mapper_to_use = mapper_user

        footprint_points_2d = [self.geo_transform * [0, 0],
                               self.geo_transform * [0, self.width],
                               self.geo_transform * [self.height, self.width],
                               self.geo_transform * [self.height, 0]]

        footprint_points_3d = mapper_to_use.map_heights_from_coordinates(footprint_points_2d, self._crs)

        if footprint_points_3d is not None:
            footprint_geom = geometry.Polygon(footprint_points_3d)
            area = float(np.round(footprint_geom.area))

            return footprint_points_3d, self._res, area

        return None

    def map_points(self, points_image: ArrayNx2,
                   mapper_user: MappingBase | None = None) -> tuple[ArrayNx3, float] | None:

        if not self.is_geo_referenced:
            raise MappingError("Image is not geo-referenced")

        if self.mapper is None and mapper_user is None:
            raise MappingError("No mapper specified")

        mapper_to_use = self.mapper
        if mapper_user is not None:
            mapper_to_use = mapper_user

        points_image = to_array_nx2(points_image)

        pt_x, pt_y = self.geo_transform * (points_image[:, 0], points_image[:, 1])

        try:
            points_3d = mapper_to_use.map_heights_from_coordinates(np.vstack((pt_x, pt_y)).T, self._crs)
        except MappingError as e:
            raise e

        if points_3d is not None:
            return points_3d, self._res

        return None
