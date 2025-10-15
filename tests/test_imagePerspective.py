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

""" Test of perspective image"""

import numpy as np
import pyproj.network
from pyproj import CRS
import pytest

from WISDAMcore.image.perspective import IMAGEPerspective
from WISDAMcore.mapping.plane import MappingPlane
from WISDAMcore.transform.rotation import Rotation
from WISDAMcore.transform.coordinates import CoordinatesTransformer
from WISDAMcore.camera.opencv_perspective import CameraOpenCVPerspective

from WISDAMcore import allow_ballpark_transformations, allow_non_best_transformations

pyproj.network.set_network_enabled(True)

@pytest.fixture
def image():
    position = np.array([478397.4630808606, -2134438.8903628434, 149.8566026660398])
    rotation = Rotation()
    rotation.opk_degree = np.array([1.4890740840939083, -0.0032084676638993266, 78.52862383108489])
    camera = CameraOpenCVPerspective(width=5472, height=3648, fx=3.75303332732689296e+03, fy=3.75303332732689296e+03,
                                     cx=2.72122638969748232e+03, cy=1.81369176603311416e+03,
                                     k1=-8.25592204996449669e-03, k2=5.34537556895088063e-04,
                                     k3=8.94905946797498221e-03,
                                     p1=-1.50822109676442082e-03, p2=-8.68606488190297281e-04)

    image = IMAGEPerspective(width=5472, height=3648, camera=camera, position=position, orientation=rotation)

    assert int(image.type)

    assert image.position_wgs84 is None
    assert image.position_wgs84_geojson is None

    assert image.crs_wkt is None
    assert image.crs_proj4 is None

    # For testing set crs after creation to test crs.setter
    image.crs = CRS("EPSG:32655+3855")

    assert image.position_wgs84
    assert image.position_wgs84_geojson

    assert image.crs_wkt
    assert image.crs_proj4

    return image


def test_local_crs():
    local_crs = CRS(r"""ENGCRS["local CRS",
      EDATUM["local datum"],
      CS[Cartesian,3],
        AXIS["(x)",east,ORDER[1]],
        AXIS["(y)",north,ORDER[2]],
        AXIS["(z)",up,ORDER[3]],
        LENGTHUNIT["metre",1.0]]""")

    reference_projection = np.array([5256.824422179638, 2331.885425631385])
    position = np.array([478397.4630808606, -2134438.8903628434, 149.8566026660398])
    rotation = Rotation()
    rotation.opk_degree = np.array([1.4890740840939083, -0.0032084676638993266, 78.52862383108489])
    camera = CameraOpenCVPerspective(width=5472, height=3648, fx=3.75303332732689296e+03, fy=3.75303332732689296e+03,
                                     cx=2.72122638969748232e+03, cy=1.81369176603311416e+03,
                                     k1=-8.25592204996449669e-03, k2=5.34537556895088063e-04,
                                     k3=8.94905946797498221e-03,
                                     p1=-1.50822109676442082e-03, p2=-8.68606488190297281e-04)

    image_local_crs = IMAGEPerspective(width=5472, height=3648, camera=camera, crs=local_crs,
                                       position=position, orientation=rotation)

    point_3d = np.array([478426.4681309458, -2134368.0357424654, 44.82127413557495])

    projection_result = image_local_crs.project(point_3d, local_crs)
    reprojected_point, valid_index = projection_result
    print("%2.1f pixel" % np.linalg.norm(reprojected_point[0, :] - reference_projection))
    assert np.linalg.norm(reprojected_point[0, :] - reference_projection) < 1.5


def test_reproject(image: IMAGEPerspective):
    # Reference image pixel point to test
    reference_projection = np.array([5256.824422179638, 2331.885425631385])

    # 3d point to test. Here image crs and 3d point crs are the same
    point_3d = np.array([478426.4681309458, -2134368.0357424654, 44.52663704111211])
    point_crs = CRS("EPSG:32655+3855")

    projection_result = image.project(point_3d, point_crs)
    reprojected_point, valid_index = projection_result
    print("%2.1f pixel" % np.linalg.norm(reprojected_point[0, :] - reference_projection))
    assert np.linalg.norm(reprojected_point[0, :] - reference_projection) < 1.5


def test_map_geometry(image: IMAGEPerspective):
    # Point in 3d which is the reference
    reference_point_3d = np.array([478426.4681309458, -2134368.0357424654, 44.52663704111211])
    point_crs = CRS("EPSG:32655+3855")

    # Picked point in image which corresponds to reference point in 3d
    reference_projection = np.array([5256.824422179638, 2331.885425631385])

    # We use the 3d point to get the height for our plane mapper
    mapper = MappingPlane(crs=point_crs, plane_altitude=reference_point_3d[2])
    image.mapper = mapper
    mapped_point, gsd = image.map_points(reference_projection)

    # The mapped point will be in the image coordinate system so we transform back to the reference point crs
    mapped_point = CoordinatesTransformer.from_crs(image.crs, point_crs, mapped_point)
    print("%2.4f m" % np.linalg.norm(mapped_point.coordinates[0, :] - reference_point_3d))
    assert np.linalg.norm(mapped_point.coordinates[0, :] - reference_point_3d) < 0.03
