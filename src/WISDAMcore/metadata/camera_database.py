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

# Todo advanced database with model and make similarity search

# Image Size version from Dugong Detector
ImageSize = {'L1D-20c': np.array([13.20, 8.80]),
             'L2D-20c': np.array([13.20, 8.80]),
             'FC3411': np.array([13.2, 8.8]),
             'DSC-RX1RM2': np.array([35.9, 24.00]),
             'ILCE-1': np.array([35.9, 24]),
             'Canon5DSR': np.array([36, 24]),
             'GFX100S': np.array([43.8, 32.9]),
             'GFX100 II': np.array([43.8, 32.9]),
             'NIKON D3200': np.array([23.2, 15.4])}


def get_sensor_from_database(make, model) -> tuple | None:
    # Find sensor in database
    sensor_width = None
    if model in ImageSize:
        sensor_width = (ImageSize[model][0], ImageSize[model][1])

    return sensor_width
