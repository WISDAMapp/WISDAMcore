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

from WISDAMcore.utils import ArrayNx3, Vector3D, Array3x3
from WISDAMcore.exceptions import RotationError


class Rotation:
    """Rotation class is used to construct a 3x3 rotation matrix
    The definitions are as follows:
    X-axis from left to right of image
    Y-axis along the y of the image with top to down
    Z-axis away from the image direction
    Global System is ENU System"""

    def __init__(self, rotation_matrix: np.ndarray | None = None) -> None:

        if rotation_matrix is None:
            rotation_matrix = np.eye(3, 3)
        self._rotation_matrix = rotation_matrix

    @classmethod
    def from_opk_degree(cls, omega_degree: float, phi_degree: float, kappa_degree: float) -> Rotation:

        rotation = cls()
        rotation.opk_degree = np.array([omega_degree, phi_degree, kappa_degree])
        return rotation

    @classmethod
    def from_opk(cls, omega: float, phi: float, kappa: float) -> Rotation:

        rotation = cls()
        rotation.opk = np.array([omega, phi, kappa])
        return rotation

    @property
    def matrix(self) -> ArrayNx3:
        """Return rotation matrix

        :return: Rotation Matrix np.array((3,3))"""
        return self._rotation_matrix

    @matrix.setter
    def matrix(self, rot: Array3x3):
        """Set rotation matrix

        :param rot: Rotation Matrix np.array((3,3))"""
        self._rotation_matrix = rot

    @property
    def opk_degree(self) -> Vector3D:
        """Omega phi kappa in degree

        :return: Rotation Omega Phi Kappa in degree - np.array(3,))"""

        return self.opk / np.pi * 180

    @property
    def opk(self):
        """convert rotation matrix to omega, phi, kappa in radians

        :return: Np Array with [Omega, Phi, Kappa] - np.array(3,))
        """
        if self._rotation_matrix.shape == (3, 3):  # convert to rotation angles ...
            omega_rad = np.arctan2(-self._rotation_matrix[1, 2], self._rotation_matrix[2, 2])

            # avoid a np domain error: argument of arcsin must be within [-1,1]
            phi_rad = np.arcsin(max(-1., min(1., self._rotation_matrix[0, 2])))
            kappa_rad = np.arctan2(-self._rotation_matrix[0, 1], self._rotation_matrix[0, 0])
            op_vec_rad = np.array([omega_rad, phi_rad, kappa_rad])

            return op_vec_rad

        raise RotationError("Rotation matrix expected, shape is: {}".format(self._rotation_matrix.shape))

    @opk_degree.setter
    def opk_degree(self, opk_vec_degree: Vector3D):
        """convert omega-, phi-, kappa-angle from Degree to rotation matrix
        """
        if opk_vec_degree.shape == (3,):
            self.opk = opk_vec_degree * np.pi / 180
        else:
            raise RotationError("Vector expected, shape is: {}".format(opk_vec_degree.shape))

    @opk.setter
    def opk(self, opk_vec_rad: Vector3D):
        """convert omega-, phi-, kappa-angle from Radians to rotation matrix
        """
        if opk_vec_rad.shape == (3,):  # convert to rotation matrix
            omega = opk_vec_rad[0]
            phi = opk_vec_rad[1]
            kappa = opk_vec_rad[2]

            som = np.sin(omega)
            com = np.cos(omega)
            sfi = np.sin(phi)
            cfi = np.cos(phi)
            ska = np.sin(kappa)
            cka = np.cos(kappa)

            # This is R, not R.T !
            opk_rot_mat = np.array([[cfi * cka, -cfi * ska, sfi],
                                    [com * ska + som * sfi * cka, com * cka - som * sfi * ska, -som * cfi],
                                    [som * ska - com * sfi * cka, som * cka + com * sfi * ska, com * cfi]])

            self._rotation_matrix = opk_rot_mat

        else:
            raise RotationError("Vector expected, shape is: {}".format(opk_vec_rad.shape))
