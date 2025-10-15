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


def intersection_plane(line_vec: np.ndarray, line_point: np.ndarray,
                       plane_point: np.ndarray, plane_normal: np.ndarray | None = None):
    epsilon = 1e-6
    # Define plane
    if plane_normal is None:
        plane_normal = np.array([0, 0, 1])

    n_dotu = plane_normal.dot(line_vec)

    if abs(n_dotu) < epsilon:
        direction_valid = False
        return np.array([0, 0, 0]), direction_valid
        # print "no intersection or line is within plane"

    w = line_point - plane_point
    si = -plane_normal.dot(w) / n_dotu
    p_si = w + si * line_vec + plane_point

    v1 = line_point - p_si
    v2 = line_point - (line_point + line_vec)
    cos_a = v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    if cos_a < 0.9:
        direction_valid = False
    else:
        direction_valid = True

    return p_si, direction_valid


def intersection_plane_mat_operation(line_vec: np.ndarray,
                                     line_point: np.ndarray,
                                     plane_point: np.ndarray,
                                     plane_normal: np.ndarray | None = None) -> np.ndarray | None:
    epsilon = 1e-6
    # Define plane
    if plane_normal is None:
        plane_normal = np.array([0, 0, 1])
    # Define ray
    # rayDirection = numpy.array([0, -1, -1])
    # rayPoint = numpy.array([0, 0, 10])  # Any point along the ray

    n_dotu = line_vec.dot(plane_normal)
    # np.einsum('ij,ij->i', plane_normal, line_vec)

    # check for parallel lines to plane
    if np.any(abs(n_dotu) < epsilon):
        return None
        # print "no intersection or line is within plane"

    w = line_point - plane_point
    si = -w.dot(plane_normal) / n_dotu

    # si * line_vec
    si_m_line_vec = np.einsum("ij,i->ij", line_vec, si)
    p_si = w + si_m_line_vec + plane_point

    # Check if the direction would be to the other direction aka backwards
    # because that is mathematically possible on a line plane intersection
    v1 = p_si - line_point
    cos_a = np.einsum('ij,ij->i', v1, line_vec) / np.linalg.norm(v1, axis=1) / np.linalg.norm(line_vec, axis=1)
    if np.any(cos_a < 0.9):
        return None

    return p_si
