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


import numpy as np
from pyproj import CRS

from WISDAMcore.transform import coordinates


def get_zone(longitude: float, latitude: float) -> int:
    """Estimate the WGS84 utm zone from latitude and longitude.

    :param longitude: Longitude in degree. Range of -180° to 180°
    :param latitude: Latitude in degree. Range -90° to 90°
    :returns: EPSG code as integer
    :raise ValueError: If limits exceeded: -180 < Longitude < 180 and -90 < Latitude < 90"""

    if abs(longitude) > 180 or abs(latitude) > 90:
        raise ValueError("Limits exceeded: -180<Longitude<180 and -90<Latitude<90")

    epsg_code_utm = int(32700 - np.round((45 + latitude) / 90, 0) * 100 + np.round((183 + longitude) / 6, 0))
    return epsg_code_utm


def point_convert_utm_wgs84_egm2008(crs_s: CRS, x: float, y: float, z: float) -> tuple | None:
    """Transform points from given CRS into WGS84-UTM coordinates and return coordinates and CRS

    :param crs_s: Coordinate reference system of the points to be converted to UTM
    :param x: X coordinate of point as float in degree
    :param y: Y coordinate of point as float in degree
    :param z: Z coordinate of point as float in meter
    :return: Tuple with coordinate(x,y,z) and CRS class. Z value is in ellipsoid heights
    :raise ValueError: WGS84 limits for utm exceeded
    :raise CoordinateConversionError: CRS, Projection or Transformation wrong"""

    try:
        coo_wgs84 = coordinates.CoordinatesTransformer.from_crs(crs_s, CRS(4979), np.array([x, y, z]))
    except:
        return None

    epsg_code_utm = get_zone(coo_wgs84.coordinates[0][0], coo_wgs84.coordinates[0][1])

    utm_crs = CRS('EPSG:' + str(epsg_code_utm) + '+3855')

    try:
        coo_utm = coo_wgs84.to_crs(utm_crs)
    except:
        return None
    x_utm, y_utm, z_geoid = coo_utm.coordinates[0]
    return x_utm, y_utm, z_geoid, utm_crs
