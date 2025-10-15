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


def get_transformation_transformergroup(crs_source, crs_target):
    """Use Transformergroup to create transformation
    !!! In the pyproj docu is stated that this returned objects are not Thread-Safe"""

    try:

        transformer = TransformerGroup(crs_source, crs_target, always_xy=True,
                                       allow_ballpark=cfg._ballpark_transformation)
        if cfg._only_best_transformation and not transformer.best_available:
            raise CoordinateTransformationError("Best Transformation is not available "
                                                "but only best transformations are allowed")
        return transformer

        # self.transformer = Transformer.from_crs(self._crs, crs_target, always_xy=True,
        #                                        only_best=cfg._only_best_transformation,
        #                                        allow_ballpark=cfg._ballpark_transformation)
    except ProjError as e:
        raise CoordinateTransformationError("Transformation could not be established.") from e


def get_transformation(crs_source, crs_target):
    try:

        transformer = TransformerFromCRS(crs_source, crs_target, always_xy=True,
                                         allow_ballpark=cfg._ballpark_transformation,
                                         always_best=cfg._only_best_transformation)

        return transformer

    except ProjError as e:
        raise CoordinateTransformationError("Transformation could not be established.") from e


class CoordinateTransformer:
    """Class to handle transformations of coordinates.
    Basically a little pyproj wrapper with basic functions needed in the package.
    """

    def __init__(self, transformer: Transformer):
        # points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2):

        """Initialize class witch coordinate reference system and points"""
        self._transformer = transformer

    @property
    def source_crs(self):
        return self._transformer.source_crs

    @property
    def target_crs(self):
        return self._transformer.target_crs

    @classmethod
    def from_crs(cls, crs_s: CRS, crs_t: CRS) -> CoordinateTransformer:
        """Create coordinate class to target coordinate system

        :param crs_s: Source coordinate reference system
        :param crs_t: Target coordinate reference system
        :returns: CoordinateTransformer class in target CRS
        :raise CoordinateConversionFailed: The conversion failed for reasons either because no transformation can be
            established or the coordinate dimension is wrong
        """

        try:
            transformer = get_transformation(crs_s, crs_t)
        except CoordinateTransformationError as e:
            raise e

        _obj = cls(transformer=transformer)

        return _obj

    def transform(self, coordinates: ArrayNx3 | ArrayNx2) -> CoordinateTransformer:
        """Transform coordinate class to target coordinate system

        :param coordinates: Coordinates to transform as ArrayNx3 for 3D and ArrayNx2 for 2D
        :returns: Coordinate class in target CRS
        :raise: CoordinateConversionFailed"""

        _coordinates = coordinates.copy()
        if _coordinates.ndim == 1:
            _coordinates = np.array([_coordinates])

        # If the same coordinate systems are used we will spare the trafo
        if self._transformer.source_crs.is_exact_same(self._transformer.target_crs):
            return _coordinates

        try:
            # TODO test errcheck
            res = self._transformer.transform(*_coordinates.T, errcheck=True)
            p_crs = np.array(res).T

            return p_crs

        except (CRSError, ProjError) as e:
            raise CoordinateTransformationError("Error while transforming coordinates") from e


class Geometries:
    pass

    def __init__(self, points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2):
        # points: ArrayNx3 | Vector3D | Vector2D | ArrayNx2):
        self._coordinates = points.copy()

        # The class uses internal only array with ndim 2. So point np.array([0,0,1]) will be np.array([[0,0,1]])
        if self._coordinates.ndim == 1:
            self._coordinates = np.array([self._coordinates])

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
