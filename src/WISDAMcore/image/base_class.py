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


from enum import Enum
from abc import abstractmethod

from pyproj import CRS

from WISDAMcore.utils import ArrayNx3, ArrayNx2, Vector3D, MaskN_
from WISDAMcore.mapping.base_class import MappingBase
from WISDAMcore.transform.coordinates import CoordinatesTransformer
#from WISDAMcore.transform.coordinates import CoordinatesTransformer, get_transformation


class ImageType(Enum):
    """Enum with the image types.
    Add new image types here as well"""

    Unknown = 0, 'unknown'
    Perspective = 1, 'perspective'
    Orthophoto = 2, 'ortho'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class ImageBase:
    """ Base class for images which is used to map pixel coordinates or project world coordinates to image
    This class holds all info geo-reference information needed for the transition.
    Already implemented are Perspective image using camera class and ortho photo imagery using Rasterio

    CRS System of image pixel: X is pointing right and y is pointing down from top left pixel corner.
    CRS Image World System can be any cartesian coordinate system.
    For mapping purpose this implementation is better to use a world crs which is a projection.
    Further developments are in plan to be able to use also geocentric coordinate system

    The basic functions within that Base Class are used to call the private functions
    which will need to be implemented.
    You will need to reimplement these functions for other image models:

        class method - "from_dict"\n

        property - "type"
                    should be according to the ImageType which should be added\n
        property - "param_dict"
                    Dictionary for which the class could be constructed using from_dict
        property - "is_geo_referenced"
                    True if the image is geo-referenced.
                    For example the perspective image position, orientation, crs and camera model must be valid
        property - "center"
                    The center of the image as you define it (e.g. imagePerspective we use principal point)
        methode - "position_to_crs"
                    Image position in WGS84 ellipsoid coordinates.
        methode - "project"
                    Project Points in the images world CRS (self._crs) to pixel
        methode - "map_center_point"
                    Map the image center point to image world CRS
        methode - "map_footprint"
                    Map footprint of image to world CRS
        methode - "map_points"
                    Map points to image world CRS

        As reference, you can have a look how in the imagePerspective and imageOrtho Class implementations
    """

    def __init__(self, width: float | int, height: float | int, mapper: MappingBase | None = None,
                 crs: CRS | None = None):

        self._crs: CRS | None = crs
        self.mapper = mapper

        self._width: int = width
        self._height: int = height

        self._position_wgs84 = None

    @property
    def mapper(self) -> MappingBase | None:
        # get mapper
        return self._mapper

    @mapper.setter
    def mapper(self, mapper: MappingBase | None):
        self._mapper = mapper
        #if self._mapper is not None and self._crs is not None:
        #    self._mapper_transformer = get_transformation(self._crs, self._mapper.crs)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict, mapper: MappingBase | None = None):
        """Initialize Class form dictionary"""
        pass

    @property
    @abstractmethod
    def type(self) -> ImageType:
        """ImageType according to class ImageType"""
        pass

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        """get image parameters as dict according to the implementation of from_dict"""
        pass

    @property
    @abstractmethod
    def is_geo_referenced(self) -> bool:
        """Returns True if image is geo-referenced and theoretically mapping can be calculated"""
        pass

    @property
    @abstractmethod
    def center(self) -> tuple:
        """Return the center of the image. (e.g. principal point for perspective cameras)"""
        pass

    @abstractmethod
    def position_to_crs(self, crs_t: CRS) -> CoordinatesTransformer | None:
        """Return the position in target CRS of the image or None if not possible.

        :returns: CoordinatesTransformer with the transformed position"""
        pass

    @abstractmethod
    def project(self, points_world_crs: ArrayNx3, crs_srs: CRS | None = None) -> tuple[ArrayNx2, MaskN_] | None:
        """Calculate projection of 3d points into image. If points was outsize image size and to_distortion is true
        the undistorted projection will be returned

        :param points_world_crs: Array of (nx3) with the point positions
        :param crs_srs: The coordinate system of the input coordinates
        :return: Tuple[Points in image space, Bool Nx1 Mask with valid pixels inside image]"""
        pass

    @abstractmethod
    def map_center_point(self, mapper_user: MappingBase | None = None) -> tuple[Vector3D, float] | None:
        """Map center point of image to 3d
        :param mapper_user: Specify Mapper to be used, can be different to the one assigned in the class
        :return: tuple[ArrayNx3 with mapped coordinates, GSD] or None if failed or no mapper specified"""
        pass

    @abstractmethod
    def map_footprint(self, mapper_user: MappingBase | None = None) -> tuple[ArrayNx3, float, float] | None:
        """Map footprint of image to 3d
        :param mapper_user: Specify Mapper to be used, can be different to the one assigned in the class
        :return: tuple[ArrayNx3 with mapped coordinates, GSD, area] or None if failed or no mapper specified"""
        pass

    @abstractmethod
    def map_points(self, points_image: ArrayNx2, mapper_user: MappingBase | None = None) -> tuple[ArrayNx3, float]:
        """Map image pixel points to 3d

        :param mapper_user: Specify Mapper to be used, can be different to the one assigned in the class
        :param points_image: Pixel coordinates
        :return: tuple[ArrayNx3 with mapped coordinates, GSD] or None if failed or no mapper specified"""
        pass

    # Standard properties
    @property
    def position_wgs84(self) -> tuple | None:
        """Return the position in WGS84 (EPSG:4979) of the image or None if not possible.
        The position_to_crs methode is defined by the class implementation.

        :returns: Tuple(x,y,z) or None"""
        coordinates = self.position_to_crs(CRS.from_epsg(4979))

        if coordinates is not None:
            pos_wgs84 = coordinates.coordinates[0]
            return pos_wgs84[0], pos_wgs84[1], pos_wgs84[2]

        return None

    @property
    def position_wgs84_geojson(self) -> dict | None:
        """Get the image position in WGS84(EPSG:4979) as geojson point or None if not geo-referenced"""
        coordinates = self.position_to_crs(CRS.from_epsg(4979))
        if coordinates is not None:
            return coordinates.geojson('Point')
        return None

    @property
    def crs(self) -> CRS:
        return self._crs

    # TODO should we actually perform a transformation?
    @crs.setter
    def crs(self, crs: CRS):
        self._crs = crs

    @property
    def crs_wkt(self) -> str | None:
        if self._crs is not None:
            return self._crs.to_wkt()
        return None

    @property
    def crs_proj4(self) -> str | None:
        if self._crs is not None:
            return self._crs.to_proj4()
        return None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def shape(self) -> tuple:
        return self._width, self._height
