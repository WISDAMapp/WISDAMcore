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
from pyproj import Transformer, CRS
from pyproj.transformer import TransformerGroup
from pyproj.exceptions import CRSError, ProjError
from functools import lru_cache
from shapely import geometry, errors as shapely_errors

from WISDAMcore.exceptions import CoordinateTransformationError

from WISDAMcore import cfg
from WISDAMcore.utils import ArrayNx3, Vector3D, Vector2D, ArrayNx2

logger = logging.getLogger(__name__)

# The doc states that cached Versions can be used to speed up performance significantly
TransformerFromCRS = lru_cache(Transformer.from_crs)
TransformerFromPipeline = lru_cache(Transformer.from_pipeline)


class CoordinatesTransformer:
    """Class which handles coordinate transformations.
    Basically a little pyproj wrapper with basic functions needed in ISDAMcore.
    The coordinates of that class are read only once initialized"""

    def __init__(self, crs: CRS, points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2):
        """Initialize class witch coordinate reference system and points"""

        self._coordinates = points.copy()

        # The class uses internal only array with ndim 2. So point np.array([0,0,1]) will be np.array([[0,0,1]])
        if self._coordinates.ndim == 1:
            self._coordinates = np.array([self._coordinates])

        # self._coordinates.setflags(write=False)
        self._crs: CRS = crs

        self.transformer = None

    @classmethod
    def from_crs(cls, crs_s: CRS,
                 crs_t: CRS, points: ArrayNx3 | Vector3D | Vector2D | ArrayNx3) -> CoordinatesTransformer:
        """Transform coordinate class to target coordinate system

        :param crs_s: Source coordinate reference system
        :param crs_t: Target coordinate reference system
        :param points: Points to be transformed
        :returns: CoordinateTransformer class in target CRS
        :raise CoordinateConversionFailed: The conversion failed for reasons either because no transformation can be
            established or the coordinate dimension is wrong
        """

        _obj = cls(crs=crs_s, points=points)

        try:
            return _obj.to_crs(crs_t)
        except CoordinateTransformationError as e:
            raise e

    @property
    def is_point(self) -> bool:
        if self._coordinates.shape[0] == 1:
            return True
        return False

    @property
    def is_3d(self):
        if self._coordinates.shape[1] == 3:
            return True
        return False

    def geojson(self, geom_type: str) -> dict:
        """Map coordinates to geojson using the specified type

        :param geom_type: String with the geometry Type.
        :returns: Geojson dict
        :raise GeoJSONGeometryNotSupported: The geometry type specified is not implemented
        :raise GeoJSONCreationFailed: The coordinates provided are not valid for that geom type (e.g. Point for Polygon)
        """

        if geom_type not in ['Point', 'LineString', 'Polygon']:
            raise CoordinateTransformationError(r"Geometry is not supported. Only 'Point', 'LineString', 'Polygon'")

        geojson = dict()
        try:

            if geom_type == 'Point':
                geojson = geometry.Point(self.coordinates)
            elif geom_type == 'LineString':
                geojson = geometry.LineString(self._coordinates)
            elif geom_type == 'Polygon':
                geojson = geometry.Polygon(self._coordinates)

        except (shapely_errors.GEOSException, ValueError) as err:
            # Will also raise if multi coordinates and point type is used
            raise CoordinateTransformationError("GeoJSON can not be established from given coordinates") from err

        geojson = geometry.mapping(geojson)

        return geojson

    @property
    def coordinates(self) -> ArrayNx3 | ArrayNx2:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, new_coordinate: ArrayNx3 | ArrayNx2):

        points = new_coordinate.copy()

        # The class uses internal only array with ndim 2. So point np.array([0,0,1]) will be np.array([[0,0,1]])
        if points.ndim == 1:
            new_coordinate = np.array([new_coordinate])
        self._coordinates = new_coordinate

    def to_crs(self, crs_target: CRS) -> CoordinatesTransformer:
        """Transform coordinate class to target coordinate system

        :param crs_target: Target coordinate reference system
        :returns: CoordinateTransformer class in target CRS
        :raise: CoordinateConversionFailed"""

        # If the same coordinate systems are used we will spare the trafo
        if self._crs.equals(crs_target):
            return CoordinatesTransformer(crs_target, self._coordinates)

        try:

            self.transformer = TransformerFromCRS(self._crs, crs_target, always_xy=True,
                                                  allow_ballpark=cfg._ballpark_transformation,
                                                  only_best=cfg._only_best_transformation)
            # if cfg._only_best_transformation and not tr_group.best_available:
            #    raise CoordinateTransformationError("Best Transformation is not available "
            #                                        "but only best transformations are allowed")
            # self.transformer = tr_group.transformers[0]

            # self.transformer = Transformer.from_crs(self._crs, crs_target, always_xy=True,
            #                                        only_best=cfg._only_best_transformation,
            #                                        allow_ballpark=cfg._ballpark_transformation)
        except ProjError as e:
            raise CoordinateTransformationError("Transformation could not be established.") from e

        try:
            if self.is_3d:
                fx, fy, fz = self.transformer.transform(self._coordinates[:, 0], self._coordinates[:, 1],
                                                        self._coordinates[:, 2],
                                                        errcheck=True)
                p_crs = np.vstack((fx, fy, fz)).T
            else:
                fx, fy = self.transformer.transform(self._coordinates[:, 0], self._coordinates[:, 1], errcheck=True)
                p_crs = np.vstack((fx, fy)).T

            new_trafo = CoordinatesTransformer(crs_target, p_crs)
            new_trafo.transformer = self.transformer
            return new_trafo

        except (CRSError, ProjError) as e:
            raise CoordinateTransformationError("Error while transforming coordinates") from e
