/*
  Copyright© 2023 Raúl Wolters(1)

  This file is part of rustronomy-core.

  rustronomy is free software: you can redistribute it and/or modify it under
  the terms of the European Union Public License version 1.2 or later, as
  published by the European Commission.

  rustronomy is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
  A PARTICULAR PURPOSE. See the European Union Public License for more details.

  You should have received a copy of the EUPL in an/all official language(s) of
  the European Union along with rustronomy.  If not, see
  <https://ec.europa.eu/info/european-union-public-licence_en/>.

  (1) Resident of the Kingdom of the Netherlands; agreement between licensor and
  licensee subject to Dutch law as per article 15 of the EUPL.
*/

use std::sync::RwLock;

use ndarray as nd;
use ndarray_rand::{
  rand_distr::{Poisson, Uniform},
  RandomExt,
};
#[cfg(feature = "plots")]
use plotters::style::RGBColor;
use rand::Rng;
use rustronomy_fits as rsf;
use rustronomy_watershed::prelude::*;

//This constant determines the randomly generated images' sizes
const RF_SIZE: (usize, usize) = (1000, 1000);

static CGPS_DATA: RwLock<Option<nd::Array3<f64>>> = RwLock::new(None);
static SMOOTH_CGPS_DATA: RwLock<Option<nd::Array3<f64>>> = RwLock::new(None);

fn get_root_path() -> std::path::PathBuf {
  const DATA_ENV: &str = "WSRS_DATA_PATH";
  let root_path =
    std::env::var(DATA_ENV).expect(&format!("enviroment variable ${DATA_ENV} not set"));
  std::path::Path::new(&root_path)
    .canonicalize()
    .expect(&format!("could not canonicalize path found in ${DATA_ENV} env. variable"))
}

static CMAP: RwLock<Vec<(u8, u8, u8)>> = RwLock::new(Vec::new());

#[inline]
#[cfg(feature = "plots")]
fn cmap(count: usize, _min: usize, _max: usize) -> Result<RGBColor, Box<dyn std::error::Error>> {
  let lock = CMAP.read()?;
  if let Some(c) = lock.get(count) {
    Ok(RGBColor(c.0, c.1, c.2))
  } else {
    //Aquire write lock
    drop(lock);
    let mut lock = CMAP.write()?;
    if lock.is_empty() {
      lock.push((0, 0, 0))
    };
    let mut rng = rand::thread_rng();
    let c = (rng.gen_range(25..u8::MAX), rng.gen_range(25..u8::MAX), rng.gen_range(25..u8::MAX));
    lock.push(c.clone());
    Ok(RGBColor(c.0, c.1, c.2))
  }
}

fn open_cgps() {
  let mut lock = CGPS_DATA.write().unwrap();

  //Check an extra time if the data is really not written yet
  if lock.is_some() {
    return;
  }

  let path = &get_root_path().join("full_cube.fits");
  let mut fits_file = rsf::Fits::open(std::path::Path::new(path)).unwrap();

  let (header, data) = fits_file.remove_hdu(0).unwrap().to_parts();
  print!("{header}");

  let array = match data.unwrap() {
    rsf::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
    _ => panic!(),
  };

  //Datacube is 3D: we hebben 2D image in 272 verschillende channels.
  let array = array.into_dimensionality().unwrap();
  lock.replace(array);
}

fn open_cgps_smooth() {
  let mut lock = SMOOTH_CGPS_DATA.write().unwrap();

  //Check an extra time if the data is really not written yet
  if lock.is_some() {
    return;
  }

  let path = &get_root_path().join("full_cube_smoothed.fits");
  let mut fits_file = rsf::Fits::open(std::path::Path::new(path)).unwrap();

  let (header, data) = fits_file.remove_hdu(0).unwrap().to_parts();
  print!("{header}");

  let array = match data.unwrap() {
    rsf::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
    _ => panic!(),
  };

  //Datacube is 3D: we hebben 2D image in 272 verschillende channels.
  let array = array.into_dimensionality().unwrap();
  lock.replace(array);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_uniform() {
  //create uniform random field
  let rf = nd::Array2::<u8>::random(RF_SIZE, Uniform::new(0, 254));

  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/uniform_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .enable_edge_correction()
    .build_merging()
    .unwrap();

  //find minima
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    rf.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(rf.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_uniform() {
  //create uniform random field
  let rf = nd::Array2::<u8>::random(RF_SIZE, Uniform::new(0, 254));

  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/uniform_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .build_segmenting()
    .unwrap();

  //find minima
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    rf.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(rf.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_poisson() {
  //create uniform random field
  let rf = nd::Array2::<f64>::random(RF_SIZE, Poisson::new(0.85f64).unwrap());

  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/poisson_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .enable_edge_correction()
    .build_merging()
    .unwrap();

  //run pre-processor and find minima
  let rf = watershed.pre_processor(rf.view());
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    rf.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(rf.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_poisson() {
  //create uniform random field
  let rf = nd::Array2::<f64>::random(RF_SIZE, Poisson::new(0.85f64).unwrap());

  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/poisson_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .build_segmenting()
    .unwrap();

  //run pre-processor and find minima
  let rf = watershed.pre_processor(rf.view());
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    rf.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(rf.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_real() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let mut lock = CGPS_DATA.read().unwrap();
  let data_cube = if lock.is_some() {
    lock.as_ref().unwrap()
  } else {
    drop(lock);
    open_cgps();
    lock = CGPS_DATA.read().unwrap();
    lock.as_ref().unwrap()
  };
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .enable_edge_correction()
    .build_merging()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_real() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let mut lock = CGPS_DATA.read().unwrap();
  let data_cube = if lock.is_some() {
    lock.as_ref().unwrap()
  } else {
    drop(lock);
    open_cgps();
    lock = CGPS_DATA.read().unwrap();
    lock.as_ref().unwrap()
  };
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .build_segmenting()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_real_with_nan() {
  //Load image -> pick slice with lots of NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let mut lock = CGPS_DATA.read().unwrap();
  let data_cube = if lock.is_some() {
    lock.as_ref().unwrap()
  } else {
    drop(lock);
    open_cgps();
    lock = CGPS_DATA.read().unwrap();
    lock.as_ref().unwrap()
  };
  let img = data_cube.slice(nd::s![.., .., 0]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/NaNreal_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .enable_edge_correction()
    .build_merging()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_real_with_nan() {
  //Load image -> pick slice with lots of NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let mut lock = CGPS_DATA.read().unwrap();
  let data_cube = if lock.is_some() {
    lock.as_ref().unwrap()
  } else {
    drop(lock);
    open_cgps();
    lock = CGPS_DATA.read().unwrap();
    lock.as_ref().unwrap()
  };
  let img = data_cube.slice(nd::s![.., .., 0]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/NaNreal_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .build_segmenting()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_gaussian() {
  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/gauß_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .enable_edge_correction()
    .build_merging()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(
    {
      println!("Loading gaussian data");
      let hdu = rsf::Fits::open(&get_root_path().join("GAUß.fits")).unwrap().remove_hdu(0).unwrap();
      match hdu.to_parts().1.unwrap() {
        rsf::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
        _ => panic!("shitties"),
      }
    }
    .into_dimensionality::<nd::Ix2>()
    .unwrap()
    .view(),
  );
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_gaussian() {
  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/gauß_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .build_segmenting()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(
    {
      println!("Loading gaussian data");
      let hdu = rsf::Fits::open(&get_root_path().join("GAUß.fits")).unwrap().remove_hdu(0).unwrap();
      match hdu.to_parts().1.unwrap() {
        rsf::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
        _ => panic!("shitties"),
      }
    }
    .into_dimensionality::<nd::Ix2>()
    .unwrap()
    .view(),
  );
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_real_smoothed() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let mut lock = SMOOTH_CGPS_DATA.read().unwrap();
  let data_cube = if lock.is_some() {
    lock.as_ref().unwrap()
  } else {
    drop(lock);
    open_cgps_smooth();
    lock = SMOOTH_CGPS_DATA.read().unwrap();
    lock.as_ref().unwrap()
  };
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_smoothed_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .enable_edge_correction()
    .build_merging()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_real_smoothed() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let mut lock = SMOOTH_CGPS_DATA.read().unwrap();
  let data_cube = if lock.is_some() {
    lock.as_ref().unwrap()
  } else {
    drop(lock);
    open_cgps_smooth();
    lock = SMOOTH_CGPS_DATA.read().unwrap();
    lock.as_ref().unwrap()
  };
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_smoothed_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::default()
    .set_plot_folder(&root)
    .set_plot_colour_map(cmap)
    .build_segmenting()
    .unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(
    img.view(),
    &root.join("original.png"),
    color_maps::viridis,
  )
  .unwrap();

  //Do transform
  watershed.transform_to_list(img.view(), mins);
}
