import pytest
from pathlib import Path
from pyproj import CRS
import numpy as np

from WISDAMcore.mapping.raster import MappingRaster, MappingType
from WISDAMcore.exceptions import MappingError

DATA_DIR = Path(__file__).parent.resolve() / "data"


@pytest.fixture(scope='session')
def raster():
    # First test wrong datafiles:
    # Test wrong path

    with pytest.raises(MappingError):
        MappingRaster(DATA_DIR / "random_jpg_wrong.JPG")

    # Test non supported file by rasterio
    with pytest.raises(MappingError):
        MappingRaster(__file__)

    # Test wrong geotransform of raster
    with pytest.raises(MappingError):
        MappingRaster(DATA_DIR / "random_jpg.JPG")

    # TODO test wrong CRS. Not sure how to do as the Mapping Raster awaits a CRS class

    print(2)
    # Test CRS system without z Axis
    with pytest.raises(MappingError):
        MappingRaster(DATA_DIR / "dhm_at_lamb_10m_2018.tif")

    mapping_raster = MappingRaster(DATA_DIR / "dhm_at_lamb_10m_2018.tif", crs=CRS("EPSG:31287+5778"))

    assert int(mapping_raster.type)

    return mapping_raster


def test_type(raster):
    assert raster.type == MappingType.Raster


def test_from_dict():
    MappingRaster.from_dict({'type': 'Raster', 'raster_filepath': (DATA_DIR / 'dhm_at_lamb_10m_2018.tif').as_posix(),
                             'crs': CRS("EPSG:31287+5778").to_wkt()})


def test_param_dict(raster):
    assert raster.param_dict == {'type': 'Raster',
                                 'raster_filepath': (DATA_DIR / 'dhm_at_lamb_10m_2018.tif').as_posix(),
                                 'crs': CRS("EPSG:31287+5778").to_wkt()}


def test_resolution(raster):
    assert np.isclose(raster.resolution, 10.004021836097209)


def test_transform(raster):
    assert False


def test_width(raster):
    assert False


def test_height(raster):
    assert False


def test_get_window(raster):
    assert False


def test_get_window_georef_array(raster):
    assert False


def test_pixel_to_coordinate(raster):
    assert False


def test_pixel_valid(raster):
    assert False


def test_coordinate_on_raster(raster):
    assert False


def test_coordinate_to_pixel(raster):
    assert False


def test_get_coordinate_height(raster):
    assert False


def test_intersection_ray(raster):
    assert False


def test_map_coordinates_from_rays(raster):
    assert False


def test_map_heights_from_coordinates(raster):
    assert False
