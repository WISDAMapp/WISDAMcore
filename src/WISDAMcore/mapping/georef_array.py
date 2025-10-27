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
from affine import Affine
from pyproj import CRS
import logging

from WISDAMcore import ArrayNx3, ArrayNx2, ArrayN_, ArrayNxN
from WISDAMcore.mapping.base_class import MappingBase, MappingType
from WISDAMcore.geometry.intersection_bilinear_patch import BilinearPatch, Vector
from WISDAMcore.transform.coordinates import CoordinatesTransformer
from WISDAMcore.exceptions import MappingError

logger = logging.getLogger(__name__)


class GeorefArray(MappingBase):
    """Class which holds the raster mapping class for WiSDAM"""

    def __init__(self, raster_array: ArrayNxN, geo_transform, crs: CRS):
        super().__init__()

        self._raster_array = raster_array
        self._raster_array.setflags(write=False)
        self._transform: Affine = geo_transform

        # User CRS can override CRS from dataset
        if crs is not None:
            self._crs = crs

        if self._crs is not None:
            if len(self._crs.axis_info) < 3:
                logger.error("CRS has no Z axis defined")
                raise MappingError("CRS has no Z axis defined")

        self._width = self._raster_array.shape[1]
        self._height = self._raster_array.shape[0]

    @property
    def transform(self):
        return self._transform

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @classmethod
    def from_dict(cls, mapper_dict: dict) -> MappingBase:
        pass

    @property
    def type(self) -> MappingType:
        return MappingType.GeoreferencedNumpyArray

    @property
    def param_dict(self) -> dict:
        pass

    def pixel_to_coordinate(self, px_row: Vector, px_col: float) -> tuple:
        """ Get coordinate in raster CRS of the pixel location
        # TODO INFO a direct matrix imput would be fine but to be sure which columns are which meaning
        # we do it like that

        :param px_row: Pixel row coordinate
        :param px_col: Pixel column coordinate
        :return: (x,y) coordinate in raster CRS
        """
        return self.transform * np.vstack((px_col, px_row))

    def coordinate_to_pixel(self, x: ArrayN_, y: ArrayN_) -> tuple:
        """ Get coordinate in raster CRS of the pixel location
        # TODO INFO a direct matrix imput would be fine but to be sure which columns are which meaning
        # we do it like that

        :param x: Pixel X coordinate
        :param y: Pixel Y coordinate
        :return: (col,row) coordinate in raster CRS
        """
        return ~self.transform * np.vstack((x, y))

    def pixel_valid(self, px_row: ArrayN_, px_col: ArrayN_) -> bool:
        """ Test if pixel coordinates is within raster limits

        :param px_row: Pixel row coordinate
        :param px_col: Pixel column coordinate
        :return: True if valid
        """
        return 0 <= px_row < self.height and 0 <= px_col < self.width

    def coordinate_on_raster(self, x_crs: ArrayN_, y_crs: ArrayN_) -> bool:
        """Test if given coordinates in raster CRS are on the raster

        :param x_crs: X coordinate
        :param y_crs: Y coordinate
        :returns: True boolean true if on the raster
        """

        pix_row, pix_col = self.coordinate_to_pixel(x_crs, y_crs)
        return self.pixel_valid(px_row=pix_row, px_col=pix_col)

    def map_coordinates_from_rays(self, ray_vectors_crs_s: ArrayNx3 | ArrayNx2, ray_start_crs_s: ArrayNx3,
                                  crs_s: CRS) -> ArrayNx3 | None:

        # TODO The length and start of the sampling could be smart estimated, as well the sampling resolution
        # TODO we could as well make the length of the sampling vector in a for loop, so to extend distance

        # establish transformer
        equal_crs = False
        if not self._crs.equals(crs_s):
            coo_trafo = CoordinatesTransformer.from_crs(crs_s, self._crs, ray_start_crs_s[0, :])
        else:
            equal_crs = True
        intersection_point = []
        for idx_ray, ray in enumerate(ray_vectors_crs_s):

            # 1. Sample the ray vectors to form an array of points.
            # The sampling is not really of importance here as here only a
            # array is loaded which does not know how big the
            # source raster could be. And its only a numpy indexing. So we use 1cm.
            dist_multi = np.arange(0, 900, 1)
            # This forms an array filled with unit vectors in the direction of the ray
            # which is than scaled by the sampled distance to get points along the ray
            ray_points_crs_s = ((np.zeros((dist_multi.shape[0], 3)) + ray / np.linalg.norm(ray)).T * dist_multi)
            # Now we translate the points to the correct location in 3d space
            ray_points_crs_s = ray_points_crs_s.T + ray_start_crs_s[idx_ray, :]

            # 2. Transform that points into the raster crs and check if all points are inside the raster
            if not equal_crs:
                ray_points_crs_array = np.array(list(coo_trafo.transformer.itransform(ray_points_crs_s, errcheck=True)))
            else:
                ray_points_crs_array = ray_points_crs_s

            # Transform position of ray points in array crs to array col and row by the affine transformation
            col, row = ~self._transform * ray_points_crs_array[:, :2].T

            # 3. Find the first approx intersection of the ray and the raster

            # Only cells, where one of the corner points is higher and one lower as the sample point,
            # are potential intersection points
            # If all corner points are lower there can be no intersection in a bilinear patch

            # TODO check if outside raster? Or at least check if an intersection was found inside the raster
            # center_cells = np.hstack((np.floor(_row)+0.5, np.floor(_col)+0.5))
            # Find unique cells which can be tested. As sampling can lead to give the same cell more times
            # unique_rows = np.unique(center_cells, axis=0)

            # TODO: What would happen if one ray point is exactly on a raster edge or line?

            # TODO: A Problem is when using the same CRS and the ray is exactly on the line
            # Then for example if line vec = [0,0.3,-1] the col lower and col upper are euqal
            # We should always get all surroundings cells not only via ceil and floor
            row_lower = np.floor(row).astype(int)
            row_upper = np.ceil(row).astype(int)
            col_lower = np.floor(col).astype(int)
            col_upper = np.ceil(col).astype(int)
            r1 = self._raster_array[row_lower, col_lower]
            r2 = self._raster_array[row_lower, col_upper]
            r3 = self._raster_array[row_upper, col_upper]
            r4 = self._raster_array[row_upper, col_lower]

            r_corners = np.vstack((r1, r2, r3, r4))

            cells_min = np.min(r_corners, axis=0)
            cells_max = np.max(r_corners, axis=0)

            # To find the indices where a possible intersection can occur,
            # 1 we check if a point is within the min an max of a cell and take this one and one before
            # Because the intersection could already happen before that point

            index_to_search = []

            index_between = np.where((cells_max >= ray_points_crs_array[:, 2]) &
                                     (ray_points_crs_array[:, 2] >= cells_min))

            if index_between[0].size != 0:
                index_to_search += index_between[0].tolist()
                # Add also the index before to it
                index_to_search += [x - 1 for x in index_between[0]]

            # 2 we test always 2 point of the ray if any of the raster values are withing that 2 points

            stack_2_points = np.vstack((ray_points_crs_array[0:-1, 2], ray_points_crs_array[1:, 2]))
            stack_2_points_max = np.max(stack_2_points, axis=0)
            stack_2_points_min = np.max(stack_2_points, axis=0)

            stack_2_cells_max = np.vstack((cells_max[0:-1], cells_max[1:]))
            stack_2_cells_max_lower = np.min(stack_2_cells_max, axis=0)

            stack_2_cells_min = np.vstack((cells_max[0:-1], cells_max[1:]))
            stack_2_cells_min_higher = np.max(stack_2_cells_min, axis=0)

            index_between = np.where((stack_2_points_max >= stack_2_cells_max_lower) &
                                     (stack_2_points_min <= stack_2_cells_min_higher))

            if index_between[0].size != 0:
                index_to_search += index_between[0].tolist()
                # Add also the index after should be added as we here get the index of the first cell
                # If we say that sampling is at least smaller than the raster sampling this is fine
                index_to_search += [x + 1 for x in index_between[0]]

            # Get unique index of cells to test with bilinear patches
            index_to_search = sorted(list(set(index_to_search)))

            # Now test all index starting from the first one till we have found an intersection
            # 4. Refine the intersection to be precise

            if len(index_to_search) > 0:

                # TODO: we could filter as well with numpy operation using r1 to r4
                #  instead of checking which cells are already looked. Should anyhow make not so much speed difference
                # We will only test cells which have not been tested. As the sampling can deliver same cells
                cells_already_looked = []

                # We now need to transform the four points into a cartesian space to run the bilinear intersection
                # TODO Maybe its better to transform all points of the for loop at once for runtime?

                for _index in index_to_search:

                    # The order of the points which perform the bilinear patch is very important!!!
                    # the 4th point should be the one on the opposite site of point 1
                    cell_corners = np.array([[row_lower[_index], col_lower[_index]],
                                             [row_upper[_index], col_lower[_index]],
                                             [row_lower[_index], col_upper[_index]],
                                             [row_upper[_index], col_upper[_index]]])

                    if cell_corners.tolist() in cells_already_looked:
                        continue

                    cells_already_looked.append(cell_corners.tolist())

                    # Get 2D pixel coordinates in raster crs
                    x, y = self.pixel_to_coordinate(px_row=cell_corners[:, 0], px_col=cell_corners[:, 1])

                    # Create 3D point from pixel coordinates and raster value (height)
                    cell_3d_points = np.vstack((x, y, self._raster_array[cell_corners[:, 0], cell_corners[:, 1]])).T

                    # TODO transformer is already established, we may should use them. Maybe rewrite coordainte class
                    # instead of doing a new CoordinateTransformer class
                    # patch_points_crs_s = CoordinatesTransformer.from_crs(self._crs, crs_s,
                    #                                                     cell_3d_points).coordinates

                    if not equal_crs:
                        patch_points_crs_s = np.array(
                            list(coo_trafo.transformer.itransform(cell_3d_points, errcheck=True, direction="inverse")))
                    else:
                        patch_points_crs_s = cell_3d_points

                    # TODO Should we check if points coplanar?
                    bilinear_patch = BilinearPatch(*patch_points_crs_s)

                    ray_vec = Vector.from_vec(ray)
                    ray_vec.normalize()

                    uv = bilinear_patch.ray_patch_intersection(pos_ray=Vector.from_vec(ray_start_crs_s[idx_ray, :]),
                                                               ray=ray_vec)

                    if uv is None:
                        uv = bilinear_patch.two_plane_approach_intersection(
                            pos_ray=Vector.from_vec(ray_start_crs_s[idx_ray, :]),
                            ray=ray_vec)

                    if uv is not None:
                        intersection_point.append(bilinear_patch.srf_eval(uv.x, uv.y).vec)
                        break

        if len(intersection_point) > 0:
            return np.array(intersection_point)
        return None

    def map_heights_from_coordinates(self, coordinates_crs_s: ArrayNx3, crs_s: CRS) -> ArrayNx3:
        pass
