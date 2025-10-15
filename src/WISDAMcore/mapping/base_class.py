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
from enum import Enum
from pyproj import CRS
from abc import abstractmethod

from WISDAMcore.utils import ArrayNx3, ArrayNx2
from WISDAMcore.exceptions import MappingError


class MappingType(Enum):
    HorizontalPlane = 1, 'horizontalPlane'
    Raster = 2, 'Raster'
    GeoreferencedNumpyArray = 2, 'GeorefNpArray'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class MappingBase:
    """Class which holds the base mapping class for WiSDAM"""

    def __init__(self):
        self._crs = None

    @classmethod
    @abstractmethod
    def from_dict(cls, mapper_dict: dict) -> MappingBase:
        """Create mapper class from dictionary"""
        pass

    @property
    @abstractmethod
    def type(self) -> MappingType:
        """Returns the mapper type as Enum MappingType"""
        pass

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        """Returns the dictionary with the mappers settings which can be used by the methode from_dict()"""
        pass

    @property
    def crs(self) -> CRS:
        """Return crs of mapper"""
        return self._crs

    @crs.setter
    def crs(self, crs: CRS):
        """Set crs of mapper"""
        self._crs = crs

    @property
    def crs_wkt(self) -> str | None:
        """Get WKT string of the CRS system is applied"""
        if self._crs is not None:
            return self._crs.to_wkt()
        return None

    @abstractmethod
    def map_coordinates_from_rays(self, ray_vectors_crs_s: ArrayNx3 | ArrayNx2, ray_start_crs_s: ArrayNx3,
                                  crs_s: CRS) -> ArrayNx3 | None:
        """Map coordinates from rays. Input vectors should be from a cartesian coordinate system.
        The coordinates of ray start should be the start of the ray as direction checks will be performed
        :param ray_vectors_crs_s: ArrayNx3 (x,y,z) with the vectors of the ray directions
        :param ray_start_crs_s: ArrayNx3 (x,y,z) with the start position of the ray vectors
        :param crs_s: The coordinate reference system of the input ray and position
        :returns: ArrayNx3(x,y,z) of the intersection points in source CRS or None if
                    mathematical no intersection could be found but inputs are valid
                  MappingError Mapping failed"""
        pass

    @abstractmethod
    def map_heights_from_coordinates(self, coordinates_crs_s: ArrayNx3, crs_s: CRS) -> ArrayNx3:
        """
        Get heights of specified coordinates. Useful to transform ortho-photo positions to 3d coordinates

        Parameters
        ----------
        coordinates_crs_s : ArrayNx3 (x,y,z)
            with the coordinates where heights should be retrieved
        crs_s : CRS
            The coordinate reference system of the input ray and position

        Returns
        -------
        Coordinates : int
             ArrayNx3(x,y,z) coordinates with the retreived heights or None if not possible
        """
        pass
