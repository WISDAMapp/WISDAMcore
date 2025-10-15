> [!WARNING]
> LEGACY-VERSION.
> This Version 0.0.1 is not any longer maintained or developed.
> It is currently still used for the current Versions of WISDAMapp
> The new version is a complete refactoring and update of the mathematically base.
> Most of the functions here are implemented long time ago where the focus was on WISDAMapp.
> Do not expect all mappings or projections are correct for high accurate geo-reference data.


# WISDAMcore
![Python Version 3.10,3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?style=flat&logo=python&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

**Python package to deal with direct geo-reference of images - MAP/PROJECT/TRANSFORM.**

It designed to simplify the use and implementation of functions and classes needed all the way from image points to mapped 
3d points or the other way around from 3d point to image points.
Additionally, it is easy to get information like mapped image footprints, center points or transform to other coordinate systems.

There are classes for camera models, images, mapping functions and additional utils which simplify and abstract
to a level where non-photogrammetry experts can work with it.

Currently, it is possible to use perspective and ortho-imagery and for mapping plane, raster (e.g. DSM) and point cloud can be used.

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [Brief History](#brief-history)
- [Goals](#goals)
- [Future Plans](#future-plans)
- [Contribution](#contribution)
- [Discussion](#discussion)
- [Package Structure](#package-structure)
- [Example Usage](#example-usage)
- [Notes on PyProj](#notes-on-pyproj)
- [License](#licence)

## Installation
The source code is currently hosted on GitHub at: https://github.com/WISDAMapp/WISDAMcore

### From Source
In the `WISDAMcore` directory (same one where you found this file after cloning the git repo), execute:
```
  pip install .
  
  # include test dependencies
  pip install .[test]
```

Testing is done using **pytest** can be found in the folder *tests*.

### Installation of 3rd-party dependencies
To provide a full simple workflow a package to read metadata is needed. see [Dependencies](#dependencies) 

## Dependencies

### Python
WISDAMcore runs on **Python 3.10+**.
WISDAMcore has been tested on Windows and Linux, and probably also runs on other Unix-like platforms.

### Packages
  It relies only on well-developed packages
- [numpy](https://www.numpy.org)
- [pyproj - Python interface to proj](https://pyproj4.github.io/pyproj/stable)
- [rasterio - Easy access to geospatial raster](https://rasterio.readthedocs.io/en/stable)
- [shapely](https://shapely.readthedocs.io/en/stable/index.html)

Additionally, to provide a full workflow the metadata from images has to be extracted.
Thus, we opted to use Phil Harvey's exiftool together with [PyExifTool](https://sylikc.github.io/pyexiftool/index.html).
Although longtime maintenance of that packages is not clear, Phil Harvey's tool provide to date the most complete metadata
reader for most image formats and physical cameras.


## Documentation
Sorry, there is currently no documentation

## Brief History
WISDAMcore was formerly part of **WISDAM** (and its predecessor DugongDetector) but is now provided as its own package.
Its used as the photogrammetry core of WISDAM to map images, objects and project them back. This allows the users to assign groups, spatial metadata and find resights.
**WISDAM (Wildlife Imagery Survey – Detection and Mapping)** is a python GUI framework based on QT for the digitization and metadata enrichment of objects
digitized in images and ortho-photos. Geo-referenced imagery can be used to map digitized objects to 3D or project them back into images for the purpose of grouping

**The WISDAM repository can be found under http://www.github.com/WISDAMapp/**

## Goals
The intention of WISDAMcore is to provide the community an easy-to-use package to deal with geo-referenced imagery.
No matter if scientific, geomatic, GIS or drone community more and more imagery is available (mostly from drones/UAS) which can be used to perform mapping operations.

As very often people are not able to use their imagery without using photogrammetric/sfm software packages to a further functionaltiy

## Future Plans
As the package is currently very basic, already a few topics are around which could be implemented:
+ Overall refactoring, especially mathematical foundation.
+ Extend image and camera model classes (For example, 360degree imagery)
+ Extend Rotation class to other common notations.
+ Import/Conversion of the results from photogrammetry / sfm packages.
+ Mapping on point cloud.
+ Mapping on mesh.
+ Provide ability to save derived geometries.
+ Maybe switch coordinate class to use geopandas.
+ Provide ability to use network assets for mapping

## Discussion
The main source of discussion should be the discussion page for discussions, questions or to get help.

## Contribution
Contributions are highly welcome to extend the package. 
All levels of contributions are welcome from extending the docs, examples, mathematical discussions, coding, test implementations, providing test samples.
Please find more info in CONTRIBUTION.md

## Package Structure

> [!IMPORTANT]  
> WISDAMcore uses pyproj and using the setting always_xy for all Transformers.

WISDAMcore was designed with flexibility and extensibility in mind.
The library consists of a few classes, each with increasingly more features.

* ``WISDAMcore.camera`` is the sub package for camera models (e.g. OpenCV camera model)
It contains only the mathematical model used to transform 2D image coordinates to 3D coordinates in the camera system
and backwards. This class mostly does not need to be called itself on a basic level but is used by the image class.
New classes can be easily implemented or extended.


* ``WISDAMcore.image`` is the subpackage which provides image classes. It is used to deal with the geo-reference
information which states the image pose in 3D space. Main functions of the classes are:
  * project: Project 3D coordinates into image 2D space
  * map_points: Map pixel points into 3D space using a provided mapper.


* ``WISDAMcore.mapping`` is the subpackage which provides mapping classes.
Currently, mapping bases on a horizontal plane (mappingPlane) and mapping based on raster (mappingRaster) using rasterio.
Main functions of the classes are:
  * map_heights_from_coordinates: Get the height of coordinates.
  * map_coordinates_from_rays: Get the intersection coordinates of a ray in 3D space


* ``WISDAMcore.transform`` is the subpackage which provides transformation classes and functions
  * cooTransformer class: Transform points using pyproj. Also provides an option to create geojson dict
  * utm_converter: Convert a coordinate to WGS84/utm and get crs class as well
  * rotation class: Class dealing with Rotation matrices used in photogrammetry

> [!INFO]
> Some additional functions in the package are needed to be used by WISDAM (e.g. the function to load classes by a dictionary)

## Example Usage

    import numpy as np
    from pyproj import CRS
    from WISDAMcore import (CameraOpenCVPerspective, IMAGEPerspective, MappingPlane)
    from WISDAMcore.transform.rotation import Rotation
    
    # We use the horizontal plane mapper in the CRS System WGS84 with EGM2008
    crs_mapper = CRS("EPSG:4326+3855")
    mapper = MappingPlane(plane_altitude=0.0, crs=crs_mapper)
    
    # Camera Model
    # The camera model's width and height is the image shape wihich was used for calibration,
    # allowing the image class to use resampled images
    cam = CameraOpenCVPerspective(width=6000, height=4000, fx=2360, fy=2360)
    
    # Image Pose
    position = np.array([602013.0, 5340384.696, 100.0])
    orientation = Rotation.from_opk_degree(omega_degree=1.5, phi_degree=5.6, kappa_degree=65.0)
    # Image Coordinage Reference System
    crs = CRS(25833)  # UTM Zone 33 
    image = IMAGEPerspective(width=6000, height=4000, mapper=mapper, camera=cam,
                             crs=crs, position=position, orientation=orientation)
    # Map to image coordinates
    result = image.map_points(np.array([[2000,300],[2300, 400]]))
    if result is not None:
      points_3d, gsd = result
      print("GSD of points: %f" % gsd)
      print("Coordinates mapped", *points_3d)
  

Refer to documentation for more examples http://scene2map.github.io/WISDAMcore/

## Notes on PyProj
This packages heavily depends on pyproj CRS and Transformer/TransformerGroups.

> [!IMPORTANT]  
> WISDAMcore uses the setting always_xy for all Transformers.

The user must take care that all data is available for the Transformation needed
([Pyproj - Datadir](https://pyproj4.github.io/pyproj/stable/api/datadir.html))  or the network option is enabled
([Pyproj - Network settings ](https://pyproj4.github.io/pyproj/stable/api/network.html#proj-network-settings)).

    pyproj.network.set_network_enabled(True)

All transformation are done by `Transformer.from_crs()`.
Standard option is to allow only best transformations and an exception will be raised otherwise (Including ballpark transformations,
[PyProj - Transformer](https://pyproj4.github.io/pyproj/stable/api/transformer.html#pyproj.transformer.Transformer.from_crs)).
To allow ballpark and allow also other than the best transformations, set:

    WISDAMcore.allow_ballpark_transformations() # sets cfg._ballpark_transformation to True
    # is the same as allow_ballpark_transformations(True), to disallow set to False
    
    WISDAMcore.allow_non_best_transformations() # sets cfg._only_best_transformation to False
    # is the same as allow_non_best_transformations(True), to disallow set to False

## Licence
WISDAMcore is licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
See the NOTICE file distributed with this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

See ``LICENSE`` for more details.

[Go to Top](#table-of-contents)