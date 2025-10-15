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


from pyproj import CRS
import numpy as np

from WISDAMcore.mapping.plane import MappingPlane


def test_from_dict():
    mapper = MappingPlane.from_dict({"plane_altitude": 43, "crs": "EPSG:4326+3855"})
    assert mapper is not None


def test_param_dict():
    mapper = MappingPlane.from_dict({"plane_altitude": 43.0, "crs": "EPSG:4979"})

    param = mapper.param_dict
    assert param == {"type": mapper.type.fullname, "plane_altitude": 43.0,
                     "plane_normal": [0, 0, 1], "crs": mapper.crs.to_wkt()}

    assert mapper.crs.equals(CRS("EPSG:4979"))


def test_map_coordinates_from_rays():
    mapper = MappingPlane.from_dict({"plane_altitude": 43.0, "crs": "EPSG:4979"})
    intersection_points = mapper.map_coordinates_from_rays(ray_start_crs_s=np.array([0, 0, 100]),
                                                           ray_vectors_crs_s=np.array([0, 0, -1]),
                                                           crs_s=CRS("EPSG:25833"))

    assert np.linalg.norm(intersection_points - np.array([0, 0, 43])) < 0.0000001


def test_map_heights_from_coordinates():
    mapper = MappingPlane.from_dict({"plane_altitude": 43.0, "crs": "EPSG:4979"})
    coordinates = np.array([602013.0, 5340384.696, 400.0])
    intersection_points = mapper.map_heights_from_coordinates(coordinates_crs_s=coordinates,
                                                              crs_s=CRS("EPSG:25833"))

    assert np.linalg.norm(intersection_points - np.array([602013.0, 5340384.696, 43.0])) < 0.00001
