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


"""Exceptions used in the package"""


class FileNotSupportedError(Exception):
    """File format is not supported for that operation"""


class CameraDictError(Exception):
    """Camera dictionary is wrong and can not be used for initialization"""


class CRSnotSpecifiedError(Exception):
    """Coordinate System is not specified or found in file"""


class CoordinateTransformationError(Exception):
    """Coordinate Conversion is not possible"""


class CRSnoZaxisError(Exception):
    """Coordinate Conversion is not possible"""


class MappingError(Exception):
    """Mapping of the geometry was not possible"""


class RotationError(Exception):
    """Mapping of the geometry was not possible"""



