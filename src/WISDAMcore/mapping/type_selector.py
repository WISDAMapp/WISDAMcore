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


from WISDAMcore.mapping.base_class import MappingType, MappingBase
from WISDAMcore.mapping.raster import MappingRaster
from WISDAMcore.mapping.plane import MappingPlane
from WISDAMcore.exceptions import MappingError


def mapper_load_from_dict(mapper_config: dict) -> MappingBase | MappingPlane | MappingRaster:
    """Load mapper from dictionary
    :param mapper_config:  Dict with the mapper configuration as specified in the mappers
    :returns: Mapper Class
    :raises ValueError: if mapper type key is not found in dictionary
    :raises NotImplementedError: If the type specified in the dictionary is not supported
    :raises FileNoFoundError: If Raster file ist not found
    :raises other error for CRS and rasterio not possible"""

    mapper_type = mapper_config.get('type', None)
    if mapper_type is None:
        raise MappingError("Mapper dictionary is not working")

    if mapper_config['type'] == MappingType.HorizontalPlane.fullname:
        mapper = MappingPlane.from_dict(mapper_config)

    elif mapper_config['type'] == MappingType.Raster.fullname:
        try:
            mapper = MappingRaster.from_dict(mapper_config)
        except MappingError as e:
            raise MappingError("Raster Mapper could not be loaded") from e
    else:
        raise MappingError("Mapper type specified is not implemented")

    return mapper
