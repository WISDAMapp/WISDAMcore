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
import logging

import numpy as np
from typing import Tuple
import math
from pathlib import Path
import affine
from shapely import box
import pyproj.exceptions
import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup
from pyproj.exceptions import CRSError, ProjError
from functools import lru_cache

from WISDAMcore.utils import ArrayNx3, Vector3D, ArrayNx2, to_array_nx2, to_array_nx3
from WISDAMcore.transform.coordinates import CoordinatesTransformer
from WISDAMcore.geometry.intersection_plane import intersection_plane
from WISDAMcore.mapping.base_class import MappingBase, MappingType
from WISDAMcore.mapping.georef_array import GeorefArray
from WISDAMcore.exceptions import MappingError, CoordinateTransformationError
from WISDAMcore import cfg

logger = logging.getLogger(__name__)

# TODO Test rasters which are mirrored or flipped by geotransform
# TODO raise error if file has more than 1 band

TransformerFromCRS = lru_cache(Transformer.from_crs)
TransformerFromPipeline = lru_cache(Transformer.from_pipeline)


class MappingRaster(MappingBase):
    """Class which holds the raster mapping class for WiSDAM"""

    def __init__(self, raster_path: Path | str, crs: CRS | None = None, allow_no_crs: bool = False,
                 preload_full_raster: bool = False, preload_window: None | tuple = None, index_band: int = 1):
        super().__init__()

        if preload_full_raster and preload_window is not None:
            raise MappingError("Preload Full raster and Preload window can not be used together")

        if preload_window:
            if len(preload_window) < 4:
                raise MappingError("Preload window needs to be a tuple or"
                                   " list with 4 items but %i are given" % len(preload_window))

        self.path = Path(raster_path)

        if not self.path.exists():
            raise MappingError("File specified not Found: %s" % self.path.as_posix())

        try:
            self._dataset: rasterio.DatasetReader = rasterio.open(self.path)
        except rasterio.errors.RasterioIOError:
            raise MappingError("File format not working for rasterio")

        if self._dataset.count > 1 and index_band < 2:
            raise MappingError("Dataset has more than 1 band but no band was specified. Band numbers start with 1")

        if index_band is not None and index_band > self._dataset.count:
            raise MappingError("Band index was specified but is higher than number of bands in raster file")

        self.index_band = index_band

        # get affine geo transform
        self._transform: affine.Affine = self._dataset.transform
        if self._transform.is_identity:
            raise MappingError("Geo Transform of raster is Identity. Probably missing or no geo-reference image")

        # RasterIO reads images in array with row is the first index if seen like a numpy array
        self._height = self._dataset.shape[0]
        self._width = self._dataset.shape[1]

        self.georef_array: GeorefArray | None = None

        # TODO this should be optimized
        if self._dataset.crs is not None:
            try:
                linear_unit_factor = self._dataset.crs.linear_units_factor
            except rasterio.errors.CRSError:
                linear_unit_factor = [1.0, 1.0]
            self._res = np.mean(self._dataset.res) * linear_unit_factor[1]

        else:
            self._res = np.mean(self._dataset.res)

        # get crs code from raster
        if self._dataset.crs is not None and crs is None:
            try:
                self._crs: CRS = CRS(self._dataset.crs)
            except pyproj.exceptions.CRSError as err:
                raise MappingError("Crs from file is not working") from err

        # User CRS can override CRS from dataset
        if crs is not None:
            self._crs = crs

        if not allow_no_crs:
            if self._crs is None:
                raise MappingError("No CRS was specified or missing in file")

            if self._crs is not None:
                if len(self._crs.axis_info) < 3:
                    logger.error("CRS has no Z axis defined")
                    raise MappingError("CRS has no Z axis defined")

    @property
    def type(self):
        return MappingType.Raster

    @classmethod
    def from_dict(cls, mapper_dict: dict) -> MappingRaster:

        try:
            crs_text = mapper_dict['crs']
            raster_path = Path(mapper_dict['raster_filepath'])
        except KeyError as err:
            raise MappingError("Crs or raster_filepath key is missing") from err

        try:
            crs = CRS.from_wkt(crs_text)
        except pyproj.exceptions.CRSError as err:
            raise MappingError("No valid CRS wkt string supported") from err

        try:
            mapping = cls(raster_path=raster_path, crs=crs)
        except MappingError as e:
            raise e
        return mapping

    @property
    def param_dict(self):
        return {"type": self.type.fullname, "raster_filepath": self.path.as_posix(), "crs": self.crs_wkt}

    @property
    def resolution(self):
        """Return the raster image resolution"""
        return self._res

    @property
    def transform(self):
        return self._transform

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def load_window(self, limits_crs: tuple | None = None):

        # TODO this is a cached operation read()
        # So it could be smart to load bigger window and reuse that raster mapper
        # Then the reading of the georef_array is without IO
        # https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html

        if not limits_crs:
            raster_array = self._dataset.read()
            return GeorefArray(raster_array=raster_array, geo_transform=self.transform, crs=self._crs)

        # Test if limits inside window
        raster_bounds = self._dataset.bounds
        shape_raster_bounds = box(*raster_bounds)
        shape_limits = box(*limits_crs)

        if not shape_raster_bounds.contains(shape_limits):
            raise MappingError("Provided window is not fully inside raster's extend")

        x_min, y_min, x_max, y_max = limits_crs

        win = self._dataset.window(x_min, y_min, x_max, y_max)

        # As the limit are within the raster extend rounding of the pixel window should be not of a problem
        # as then row and col can only be >= 0.0
        # For numerical reasos (not sure if this can ever happen) we round to 0 everything lower than 1.0
        win_col_offset_rounded = np.floor(win.col_off)
        if win.col_off < 0:
            win_col_offset_rounded = 0
        win_row_offset_rounded = np.floor(win.row_off)
        if win.row_off < 0:
            win_row_offset_rounded = 0

        col_max = np.ceil(win.col_off + win.width)
        row_max = np.ceil(win.row_off + win.height)
        width = col_max - win_col_offset_rounded + 1
        height = row_max - win_row_offset_rounded + 1
        rounded_win = Window(col_off=win_col_offset_rounded, row_off=win_row_offset_rounded,
                             width=width, height=height)
        affine_of_window = self._dataset.window_transform(rounded_win)

        self.georef_array = GeorefArray(raster_array=self._dataset.read(indexes=self.index_band,
                                                                        window=rounded_win),
                                        geo_transform=affine_of_window, crs=self._crs)

    def pixel_to_coordinate(self, px_row: float, px_col: float) -> tuple:
        """ Get coordinate in raster CRS of the pixel location

        :param px_row: Pixel row coordinate
        :param px_col: Pixel column coordinate
        :return: (x,y) coordinate in raster CRS
        """
        return self.transform * (px_col, px_row)

    def pixel_valid(self, px_row: float, px_col: float) -> bool:
        """ Test if pixel coordinates is within raster limits

        :param px_row: Pixel row coordinate
        :param px_col: Pixel column coordinate
        :return: True if valid
        """
        return 0 <= px_row < self.height and 0 <= px_col < self.width

    def coordinate_on_raster(self, x_crs: float, y_crs: float) -> bool:
        """Test if given coordinates in raster CRS are on the raster

        :param x_crs: X coordinate
        :param y_crs: Y coordinate
        :returns: True boolean true if on the raster
        """

        pix_row, pix_col = self.coordinate_to_pixel(x_crs, y_crs)
        return self.pixel_valid(px_row=pix_row, px_col=pix_col)

    def coordinate_to_pixel(self, x_crs: float, y_crs: float) -> tuple[float, float]:
        """Coordinate to pixel

        :param x_crs: X coordinate
        :param y_crs: Y coordinate
        :returns: Array Like (1x2) with raster pixel values
        """
        pix_col, pix_row = ~self._transform * (x_crs, y_crs)

        return pix_row, pix_col

    def get_coordinate_height(self, x_crs: float, y_crs: float) -> float:
        """ Get raster value of coordinates in Raster CRS coordinates

        :param x_crs: X coordinate
        :param y_crs: Y coordinate
        :returns: Raster value or 0 if not valid"""
        # Check if raster coordinates are valid
        for val in self._dataset.sample([(x_crs, y_crs)]):
            height = val[0]
            if height in self._dataset.get_nodatavals():
                # if math.isclose(height, self._dataset.nodata):
                return 0
            return height

    def intersection_ray(self,
                         ray_vector_crs_s: Vector3D,
                         ray_start_crs_s: Vector3D,
                         crs_s: CRS, max_iter: int = 20) -> Tuple[Vector3D, bool, bool]:
        """ Calculate intersection of a ray with the raster
        The position on the ray should be on the raster as well, its position will be used
        to retrieve the start height of the iteration from the raster

        :param ray_vector_crs_s: The ray(vector) used for intersection, Array 1x3 in the coordinates source srs
        :param ray_start_crs_s: Start Point of the ray, Array 1x3 in the coordinates source srs
        :param crs_s: Coordinate System of the ray and ray position point
        :param max_iter: Maximum number of iterations before we exit
        :returns: Tuple (intersection point 1x3, validity of the intersection, flag if new point is outside raster)"""

        intersect_coo = np.array([0, 0, 0])

        # We could use self.map_heights_from_coordinates but for speed we will
        # only initialize a coordinateTransformer once
        # initial height

        # There seems to be an issue with PYPROJ Transformers, so we make an intermediate step to EPSG:4979

        try:
            transformer_srs_4979 = TransformerFromCRS(crs_s, "EPSG:4979", always_xy=True,
                                                      allow_ballpark=cfg._ballpark_transformation,
                                                      only_best=cfg._only_best_transformation)

            transformer_4979_mapper = TransformerFromCRS("EPSG:4979", self.crs, always_xy=True,
                                                         allow_ballpark=cfg._ballpark_transformation,
                                                         only_best=cfg._only_best_transformation)
        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        # tr_group = TransformerGroup(crs_s, self._crs, always_xy=True,
        #                            allow_ballpark=cfg._ballpark_transformation,
        #                            )
        # if cfg._only_best_transformation and not tr_group.best_available:
        #    raise MappingError("Best Transformation is not available but only best transformations are allowed")
        # tr_crs_s_to_self_crs = tr_group.transformers[0]

        try:
            plane_point_4979 = transformer_srs_4979.transform(ray_start_crs_s[0], ray_start_crs_s[1],
                                                              ray_start_crs_s[2])
            if np.any(plane_point_4979 == np.inf):
                raise MappingError("Coordinate Transformation of Mapping failed")
            plane_point_mapper_crs = transformer_4979_mapper.transform(*plane_point_4979)
            if np.inf in plane_point_mapper_crs:
                raise MappingError("Coordinate Transformation of Mapping failed")

        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        height = self.get_coordinate_height(x_crs=plane_point_mapper_crs[0], y_crs=plane_point_mapper_crs[1])

        try:
            plane_point_4979 = transformer_4979_mapper.transform(plane_point_mapper_crs[0],
                                                                 plane_point_mapper_crs[1], height, direction="inverse")

            plane_point_source_crs = transformer_srs_4979.transform(*plane_point_4979, direction="inverse")

        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        height_iteration = ray_start_crs_s[2] / 2.0 + plane_point_source_crs[2] / 2.0

        count_iteration = 0
        height_transformed = 0

        # Iterate till the difference is smaller than 0.02meter or 20 times no
        # height was found even though we are on the raster (maybe hole)
        while height_iteration - height_transformed > 0.02 and count_iteration < 20:

            # is also a point [0,0,height_iteration] working?
            plane_point = np.array([ray_start_crs_s[0], ray_start_crs_s[1], height_iteration])

            # get the intersection point
            intersect_coo, valid_intersection = intersection_plane(ray_vector_crs_s, ray_start_crs_s, plane_point)

            # check if the direction is not backwards
            if not valid_intersection:
                return intersect_coo, False, False

            # new Height from the estimated position
            # plane_point_mapper_crs = tr_crs_s_to_self_crs.transform(intersect_coo[0], intersect_coo[1],
            #                                                        intersect_coo[2])

            try:
                plane_point_4979 = transformer_srs_4979.transform(intersect_coo[0],
                                                                  intersect_coo[1], intersect_coo[2])

                plane_point_mapper_crs = transformer_4979_mapper.transform(*plane_point_4979)

            except (CRSError, ProjError) as e:
                raise MappingError("Coordinate Transformation of Mapping failed") from e

            if self.coordinate_on_raster(plane_point_mapper_crs[0], plane_point_mapper_crs[1]):
                height_pixel = self.get_coordinate_height(plane_point_mapper_crs[0], plane_point_mapper_crs[1])
                count_iteration += 1

                try:
                    plane_point_4979 = transformer_4979_mapper.transform(plane_point_mapper_crs[0],
                                                                         plane_point_mapper_crs[1],
                                                                         height_pixel, direction="inverse")

                    _, _, height_transformed = transformer_srs_4979.transform(*plane_point_4979, direction="inverse")

                except (CRSError, ProjError) as e:
                    raise MappingError("Coordinate Transformation of Mapping failed") from e
                # _, _, height_transformed = tr_crs_s_to_self_crs.transform(plane_point_mapper_crs[0],
                #                                                          plane_point_mapper_crs[1],
                #                                                          height_pixel, direction="inverse")
            else:
                return intersect_coo, False, True

            # This is to be sure that we will not run in a loop
            height_iteration = (height_iteration * 0.1 + height_transformed * 0.9)

            intersect_coo = np.array([intersect_coo[0], intersect_coo[1], height_transformed])

        if count_iteration >= max_iter:
            return intersect_coo, False, False

        return intersect_coo, True, False

    def map_coordinates_from_rays(self, ray_vectors_crs_s: ArrayNx3, ray_start_crs_s: ArrayNx3,
                                  crs_s: CRS) -> ArrayNx3:

        # We will transform the coordinate of the raster crs with the height to the image crs for intersection
        # So that we do not need to transform the ray vector according to the projection
        # It is for sure slower than transforming the ray vector one time but this is easier to implement
        # As well with that approach the raster can be geographic as well without the need to translate a raster subset

        list_coo = []

        for idx, coo in enumerate(ray_start_crs_s):
            res = self.intersection_ray(ray_vectors_crs_s[idx, :], coo, crs_s=crs_s)
            inter_p, valid_flag, outside_flag = res

            if outside_flag:
                raise MappingError("Intersection is outside of provided raster")
            if not valid_flag:
                raise MappingError("Intersection of ray is in wrong direction")
            list_coo.append(inter_p)

        coo_crs_source = np.array(list_coo)
        return coo_crs_source

    def map_heights_from_coordinates(self, coordinates_crs_s: ArrayNx3 | ArrayNx2, crs_s: CRS) -> ArrayNx3 | None:

        coordinates_crs_s = to_array_nx3(coordinates_crs_s)

        # Convert Nx2 array to Nx3
        if coordinates_crs_s.shape[1] < 3:
            coordinates_crs_s = np.hstack((coordinates_crs_s, np.zeros((coordinates_crs_s.shape[0], 1))))

        # TODO could that be of a problem if a 3d proection is used?

        try:
            # coo_crs_mapper = CoordinatesTransformer.from_crs(crs_s, self._crs, coordinates_crs_s)
            # except CoordinateTransformationError:
            # raise MappingError("Coordinate Transformation of Mapping failed")

            transformer_srs_4979 = TransformerFromCRS(crs_s, "EPSG:4979", always_xy=True,
                                                      allow_ballpark=cfg._ballpark_transformation,
                                                      only_best=cfg._only_best_transformation)

            transformer_4979_mapper = TransformerFromCRS("EPSG:4979", self.crs, always_xy=True,
                                                         allow_ballpark=cfg._ballpark_transformation,
                                                         only_best=cfg._only_best_transformation)

            fx, fy, fz = transformer_srs_4979.transform(coordinates_crs_s[:, 0], coordinates_crs_s[:, 1],
                                                        coordinates_crs_s[:, 2])

            if np.any(fx == np.inf) or np.any(fy == np.inf) or np.any(fz == np.inf):
                raise MappingError("Coordinate Transformation of Mapping failed")

            fx1, fy1, fz1 = transformer_4979_mapper.transform(fx, fy, fz)

            if np.any(fx1 == np.inf) or np.any(fy1 == np.inf) or np.any(fz1 == np.inf):
                raise MappingError("Coordinate Transformation of Mapping failed")


            points_mapper_crs = np.array([fx1, fy1, fz1]).T

        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        for idx, coo in enumerate(points_mapper_crs):
            height = self.get_coordinate_height(x_crs=coo[0], y_crs=coo[1])
            points_mapper_crs[idx, 2] = height

        try:
            # coo_crs_source = coo_crs_mapper.to_crs(crs_s).coordinates
            fx2, fy2, fz2 = transformer_4979_mapper.transform(points_mapper_crs[:, 0], points_mapper_crs[:, 1],
                                                              points_mapper_crs[:, 2], direction="inverse")
            fx3, fy3, fz3 = transformer_srs_4979.transform(fx2, fy2, fz2, direction="inverse")

            coo_crs_source = np.array([fx3, fy3, fz3]).T
        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        # except CoordinateTransformationError:
        #    raise MappingError("Coordinate Transformation of Mapping failed")

        return coo_crs_source
