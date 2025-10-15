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
from functools import lru_cache
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError, ProjError

from WISDAMcore.utils import ArrayNx3, ArrayNx2, Vector3D
from WISDAMcore.geometry.intersection_plane import intersection_plane_mat_operation
from WISDAMcore.mapping.base_class import MappingBase, MappingType
from WISDAMcore.transform.coordinates import CoordinatesTransformer
from WISDAMcore.exceptions import MappingError, CoordinateTransformationError
from WISDAMcore import cfg

TransformerFromCRS = lru_cache(Transformer.from_crs)
TransformerFromPipeline = lru_cache(Transformer.from_pipeline)


class MappingPlane(MappingBase):
    """Class dealing with mapping on plane defined by a plane point and a plane normal
    It is important to specify a correct plane point even for a horizontal plane because we will transform"""

    def __init__(self, crs: CRS | None = None,
                 plane_altitude: float | int = 0.0,
                 plane_normal: Vector3D | None = None,
                 standard_crs: bool = False,
                 allow_no_crs: bool = False):
        super().__init__()

        self.plane_altitude = float(plane_altitude)
        self.plane_normal = plane_normal
        if plane_normal is None:
            self.plane_normal = np.array([0, 0, 1])

        # If no CRS is specified we need to assume that one is in wgs84 geoid
        if crs is not None:
            self._crs = crs

        if standard_crs and self._crs is None:
            self._crs = CRS("EPSG:4326+3855")

    @classmethod
    def from_dict(cls, mapper_dict: dict) -> MappingPlane | None:

        if mapper_dict:
            plane_altitude = mapper_dict.get('plane_altitude', 0.0)
            plane_normal = mapper_dict.get('plane_normal', None)
            if plane_normal is not None:
                plane_normal = np.array(plane_normal)
            crs_text = mapper_dict.get('crs', None)

            crs = None
            if crs_text:
                crs = CRS(crs_text)

            mapping = cls(plane_altitude=plane_altitude, plane_normal=plane_normal, crs=crs)
            return mapping
        return None

    @property
    def type(self):
        return MappingType.HorizontalPlane

    @property
    def param_dict(self):
        plane_normal = self.plane_normal.tolist()
        return {"type": self.type.fullname,
                "plane_altitude": self.plane_altitude,
                "plane_normal": plane_normal,
                "crs": self.crs_wkt}

    def map_coordinates_from_rays(self, ray_vectors_crs_s: ArrayNx3, ray_start_crs_s: ArrayNx3,
                                  crs_s: CRS) -> ArrayNx3:

        # A Array(3,) should also work
        ray_pos = np.array(ray_start_crs_s)
        if ray_pos.ndim == 1:
            ray_pos = np.array([ray_pos])

        # A Array(3,) should also work
        ray_vec = np.array(ray_vectors_crs_s)
        if ray_vec.ndim == 1:
            ray_vec = np.array([ray_vec])

        if ray_pos.shape != ray_vec.shape:
            raise MappingError("Ray and Pos Vector not the same size")

        # transform the given coordinates to the mappers crs
        # replace the z coordinate of the coordinates in mapper crs with the mappers height
        # transform back to the source CRS which now provides mapper height in source crs
        # Therefore the mapper's height is correct transformed to the image crs at the image's prj center
        #try:
        #    coo_crs_mapper = CoordinatesTransformer.from_crs(crs_s, self._crs, ray_pos)
        #except CoordinateTransformationError as e:
        #    raise MappingError("Coordinate Transformation error") from e


        try:
            transformer_srs_4979 = TransformerFromCRS(crs_s, "EPSG:4979", always_xy=True,
                                                      allow_ballpark=cfg._ballpark_transformation,
                                                      only_best=cfg._only_best_transformation)

            transformer_4979_mapper = TransformerFromCRS("EPSG:4979", self._crs, always_xy=True,
                                                         allow_ballpark=cfg._ballpark_transformation,
                                                         only_best=cfg._only_best_transformation)
        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e


        try:
            fx, fy, fz = transformer_srs_4979.transform(ray_pos[:, 0], ray_pos[:, 1],
                                                        ray_pos[:, 2])

            fx1, fy1, fz1 = transformer_4979_mapper.transform(fx, fy, fz)

            plane_point_mapper_crs = np.array([fx1, fy1, fz1]).T

            # Now change the altitude to the plane height
            plane_point_mapper_crs[:, 2] = self.plane_altitude

        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        # coo_crs_mapper.coordinates[:, 2] = self.plane_altitude

        try:
            # coo_crs_source = coo_crs_mapper.to_crs(crs_s).coordinates
            fx2, fy2, fz2 = transformer_4979_mapper.transform(plane_point_mapper_crs[:, 0], plane_point_mapper_crs[:, 1],
                                                              plane_point_mapper_crs[:, 2], direction="inverse")
            fx3, fy3, fz3 = transformer_srs_4979.transform(fx2, fy2, fz2, direction="inverse")

            plane_point_source_crs = np.array([fx3, fy3, fz3]).T
        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e


        # plane_point_source_crs = coo_crs_mapper.to_crs(crs_s).coordinates

        # We also will transform the plane normal as this is also affected by coordinate transformations
        # We will just use a point on the plane normal about 100meters away.
        # Now this will get a bit complicated as we need to know if these are spherical coordinates or cartesian
        # We could do a little trick and transform first to ECEF coordinates.
        # Might be slower but more robust no matter which system is given for the plane
        intersect_points = intersection_plane_mat_operation(ray_vec, ray_pos,
                                                            plane_point=plane_point_source_crs,
                                                            plane_normal=self.plane_normal)
        if intersect_points is None:
            raise MappingError("Rays are either parallel to plane or direction of intersection is wrong")
        return intersect_points

    def map_heights_from_coordinates(self, coordinates_crs_s: ArrayNx3 | ArrayNx2, crs_s: CRS) -> ArrayNx3:

        # Prepare input if something else than ArrayNx3 (e.g. List, ArrayN)
        coordinates = np.array(coordinates_crs_s)
        if coordinates.ndim == 1:
            coordinates = np.array([coordinates])

        # Make 3D array out of 2D coordinates
        if coordinates.shape[1] < 3:
            coordinates = np.hstack((coordinates, np.zeros((coordinates.shape[0], 1))))

        try:
            transformer_srs_4979 = TransformerFromCRS(crs_s, "EPSG:4979", always_xy=True,
                                                      allow_ballpark=cfg._ballpark_transformation,
                                                      only_best=cfg._only_best_transformation)

            transformer_4979_mapper = TransformerFromCRS("EPSG:4979", self._crs, always_xy=True,
                                                         allow_ballpark=cfg._ballpark_transformation,
                                                         only_best=cfg._only_best_transformation)
        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        try:
            fx, fy, fz = transformer_srs_4979.transform(coordinates[:, 0], coordinates[:, 1],
                                                        coordinates[:, 2])

            fx1, fy1, fz1 = transformer_4979_mapper.transform(fx, fy, fz)

            plane_point_mapper_crs = np.array([fx1, fy1, fz1]).T

            # Now change the altitude to the plane height
            plane_point_mapper_crs[:, 2] = self.plane_altitude

        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        # coo_crs_mapper.coordinates[:, 2] = self.plane_altitude

        try:
            # coo_crs_source = coo_crs_mapper.to_crs(crs_s).coordinates
            fx2, fy2, fz2 = transformer_4979_mapper.transform(plane_point_mapper_crs[:, 0],
                                                              plane_point_mapper_crs[:, 1],
                                                              plane_point_mapper_crs[:, 2], direction="inverse")
            fx3, fy3, fz3 = transformer_srs_4979.transform(fx2, fy2, fz2, direction="inverse")

            plane_point_source_crs = np.array([fx3, fy3, fz3]).T
        except (CRSError, ProjError) as e:
            raise MappingError("Coordinate Transformation of Mapping failed") from e

        return plane_point_source_crs
