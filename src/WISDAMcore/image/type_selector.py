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


from WISDAMcore.image.base_class import ImageType, ImageBase
from WISDAMcore.image.ortho import IMAGEOrtho
from WISDAMcore.image.perspective import IMAGEPerspective
from WISDAMcore.mapping.base_class import MappingBase


def get_image_from_dict(param_dict: dict, mapper: MappingBase | None = None) -> ImageBase | None:
    """Get image class from dict as received by image.param_dict

    :param param_dict: Dictionary with the images parameter
    :param mapper: Mapping class. Optional
    :return: Image class or None if not possible
    """

    if param_dict.get('type') is None:
        return None

    if param_dict['type'] == ImageType.Orthophoto.fullname:
        image = IMAGEOrtho.from_dict(param_dict, mapper)

    elif param_dict['type'] == ImageType.Perspective.fullname:
        image = IMAGEPerspective.from_dict(param_dict, mapper)

    else:
        return None

    return image
