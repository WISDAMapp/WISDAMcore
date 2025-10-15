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


# https://imagemagick.org/Usage/lens/correcting_lens_distortions.pdf

"""This is designed to be used by exiftool or wrapper
   Pyexif uses different meta tags"""

import logging
import math

from WISDAMcore.metadata.camera_database import get_sensor_from_database

# Unit conversion factor
inch_to_mm = 25.4
cm_to_mm = 10
um_to_mm = 0.001

logger = logging.getLogger(__name__)


def get_image_dimensions(meta_data: dict) -> tuple[int, int] | None:
    """Get image width and height in pixel

    :param meta_data: Metadata dict of image following exiftool tag names
    :returns: Tuple(width, height) or None if failed"""

    if "EXIF:ImageWidth" in meta_data.keys():
        width = int(meta_data["EXIF:ImageWidth"])
        height = int(meta_data["EXIF:ImageHeight"])
    elif "File:ImageWidth" in meta_data.keys():
        width = int(meta_data["File:ImageWidth"])
        height = int(meta_data["File:ImageHeight"])
    elif "XMP:ImageWidth" in meta_data.keys():
        width = int(meta_data["XMP:ImageWidth"])
        height = int(meta_data["XMP:ImageHeight"])
    else:
        return None

    return width, height


def get_unit_factor(resolution_unit) -> float | None:
    """Factor to scale the exif resolution unit to millimeter which is used for Focal length in EXIF.
    Tag 0xa20e - FocalPlaneXResolution
    Tag 0xa210 - FocalPlaneResolutionUnit
    https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
    We assume square Pixels, so we only will do it for the image width side

    :param resolution_unit: the resolution unit value given in the EXIF
    :return: Unit factor or None
    """
    if resolution_unit == 2:  # Inch
        return inch_to_mm
    elif resolution_unit == 3:  # Centimeter
        return cm_to_mm
    elif resolution_unit == 4:  # Millimeter
        return 1
    elif resolution_unit == 5:  # Micrometer
        return um_to_mm
    else:
        return None


def compute_sensor_width_in_mm(image_width: int, meta_data: dict) -> float | None:
    """Get Sensor with in mm using the image width in pixel and the tags ResolutionUnit and FocalPlaneXResolution.

       :param image_width: Width of image in Pixels
       :param meta_data: Dictionary with the EXIF/XMP metadata following exiftool tag names
       :returns: Sensor width in mm or None if not possible"""

    if ("EXIF:FocalPlaneResolutionUnit" not in meta_data.keys() or
            "EXIF:FocalPlaneXResolution" not in meta_data.keys()):
        # Metadata is not holding the needed info
        return None

    resolution_unit = meta_data["EXIF:FocalPlaneResolutionUnit"]
    unit_factor = get_unit_factor(resolution_unit)

    if not unit_factor:
        return None

    pixels_per_unit = meta_data.get("EXIF:FocalPlaneXResolution", None)
    pixels_per_unit_y = meta_data.get("EXIF:FocalPlaneYResolution", None)

    if pixels_per_unit is None or pixels_per_unit_y is None:
        return None

    if pixels_per_unit <= 0.0:

        # Some wrongly formatted camera have negative resolutions
        # We check if at leas YResolution is present and if not negative use that
        if pixels_per_unit_y is None or pixels_per_unit_y <= 0.0:
            return None
        pixels_per_unit = pixels_per_unit_y

    pixel_pitch = unit_factor / pixels_per_unit

    return image_width * pixel_pitch


def compute_focal_length_from_35mm(focal_35mm: float,
                                   image_width: int | float,
                                   image_height: int | float) -> float | None:
    # If a focal length of 35mm exists together with image width this is a good approximating
    if focal_35mm > 0:
        mm_to_pixel = 43.3 / math.sqrt(image_width**2 + image_height**2)  # 35mm film have a sensor size of 36x24mm.
        focal_pixel = focal_35mm / mm_to_pixel
        return focal_pixel
    return None


def compute_from_sensor_width(focal_mm: float, sensor_width: float, image_width: int) -> float:
    return focal_mm / (sensor_width / image_width)


def estimate_camera(meta_data: dict) -> tuple | None:
    """Estimate the standard camera parameters.
       Even though I am not sure if 35mm equivalent is always calculated correctly we will use that as first source,
       because for resampled images there could be that original exif-tags are left (e.g. PlaneResolution for example)
       which would be wrong for the resampled image to have correct focal length in pixel.
       Currently, 4 possible ways are implemented
       (1) Using 35mm equivalent
       (2) Using focal length and Resolution unit
       (4) Using focal length and Sensor Database; this works only if camera type is present in database

        More advanced calibration tags in XMP like that one from Pix4D will be treated separately

    :param meta_data: Metadata dict following exif-tools tags
    :returns tuple(width, height, focal_pixel, c_x, c_y)
                width, height of image
                focal length in pixel
                c_x, c_y are the principal point - only rare found in standard exif tags"""

    # Get dimensions of image
    dimension = get_image_dimensions(meta_data=meta_data)
    if dimension is None:
        return None

    width, height = dimension

    make = meta_data.get("EXIF:Make", '')
    model = meta_data.get("EXIF:Model", '')

    # FocalLength is the real focal length in mm
    focal = meta_data.get("EXIF:FocalLength", 0.0)

    # Focal_35mm is the focal length if the camera would be a 35mm Format camera to have the same field of view
    # If focal_35 is present it is always possible to derive focal length in pixel
    # Therefore we try this first as then we are always able to create a valid camera class
    # But as it turns out, not all vendors are very precise how this is derived compared to the focal length
    focal_35 = meta_data.get('EXIF:FocalLengthIn35mmFormat', 0.0)

    focal_pixel = None
    focal_pixel_list = []

    # 1st try - get focal length in pixel from 35mm equivalent
    if focal_35 > 0:
        focal_pixel = compute_focal_length_from_35mm(focal_35, width, height)
        focal_pixel_list.append(focal_pixel)
        logger.debug("Estimated Focal Length in pixel from 35mm equivalent: f=%6.2f" % focal_pixel)

    # 2nd try  - estimate focal length in pixel from the tag focalLength
    # We will always try this as well even if focal_35 is present
    if focal > 0.0:

        sensor_width = compute_sensor_width_in_mm(width, meta_data)
        if sensor_width is not None:
            focal_pixel = compute_from_sensor_width(focal, sensor_width=sensor_width, image_width=width)
            focal_pixel_list.append(focal_pixel)
            logger.debug("Estimated Focal Length in pixel from sensor with: f=%6.2f" % focal_pixel)
            
    if len(focal_pixel_list) == 2:
        focal_pixel = (focal_pixel_list[0] + focal_pixel_list[1]) / 2.0
    # Principal point default will be center of image size
    # Later we try if in the metadata are better values
    c_x = width / 2.0
    c_y = height / 2.0

    # 3rd try - We check other tags from different software packages and drone vendors
    # E.g. newer DJI model save calibrated values CalibratedFocalLength, CalibratedOpticalCenterX/Y
    # Or Pix4D has specified a tag system for the IOR and also for coordinate systems
    # CalibratedFocalLength: 3666.666504
    # CalibratedOpticalCenterX: 2736.000000

    # We use a wrapper to call all different tag systems
    # res = calibration_estimator(meta_data)
    # if res is not None:
    #    width, height, focal_pixel, c_x, c_y, distortion = res

    # 4th and last try - estimate focal length from sensor database -
    # No sensor size is available in metadata
    # This is highly depending on the camera database and if the sensor is available there
    # If the sensor is not found you can add it in "camare_database.py"
    # Anyhow we will try this as this might be more accurate then estimating sensor size from exif data
    if focal_pixel is None:
        if focal > 0.0:
            sensor_size = get_sensor_from_database(make=make, model=model)
            if sensor_size is not None:
                focal_pixel = compute_from_sensor_width(focal, sensor_width=sensor_size[0], image_width=width)
                # logger.debug("Estimated Focal Length in pixel from sensor with from database: f=%6.2f" % focal_pixel)

    return width, height, focal_pixel, c_x, c_y
