/*
  Copyright© 2022 Raúl Wolters(1)

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

use ndarray as nd;
use ndarray_rand::{
  rand_distr::{Poisson, Uniform},
  RandomExt,
};
use rustronomy_fits as rsf;
use rustronomy_watershed::prelude::*;

//This constant determines the randomly generated images' sizes
const RF_SIZE: (usize, usize) = (1000, 1000);

fn get_root_path() -> std::path::PathBuf {
  const DATA_ENV: &str = "WSRS_DATA_PATH";
  let root_path =
    std::env::var(DATA_ENV).expect(&format!("enviroment variable ${DATA_ENV} not set"));
  std::path::Path::new(&root_path)
    .canonicalize()
    .expect(&format!("could not canonicalize path found in ${DATA_ENV} env. variable"))
}

fn open_image(path: &std::path::Path) -> nd::Array<f64, nd::Ix3> {
  let mut fits_file = rsf::Fits::open(std::path::Path::new(path)).unwrap();

  let (header, data) = fits_file.remove_hdu(0).unwrap().to_parts();
  print!("{header}");

  let array = match data.unwrap() {
    rsf::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
    _ => panic!(),
  };

  //Datacube is 3D: we hebben 2D image in 272 verschillende channels.
  array.into_dimensionality().unwrap()
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
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //find minima
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(rf.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(rf.view(), mins);
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
  let watershed = TransformBuilder::new_segmenting(&root).build().unwrap();

  //find minima
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(rf.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(rf.view(), mins);
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
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let rf = watershed.pre_processor(rf.view());
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(rf.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(rf.view(), mins);
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
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let rf = watershed.pre_processor(rf.view());
  let mins = &watershed.find_local_minima(rf.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(rf.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(rf.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_real() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let data_cube = open_image(&root.join("full_cube.fits"));
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_real() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let data_cube = open_image(&root.join("full_cube.fits"));
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_real_with_nan() {
  //Load image -> pick slice with lots of NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let data_cube = open_image(&root.join("full_cube.fits"));
  let img = data_cube.slice(nd::s![.., .., 0]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/NaNreal_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_real_with_nan() {
  //Load image -> pick slice with lots of NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let data_cube = open_image(&root.join("full_cube.fits"));
  let img = data_cube.slice(nd::s![.., .., 0]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/NaNreal_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_gaussian() {
  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/gauß_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

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
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_gaussian() {
  //make output folder and configure the watershed transform
  let root = get_root_path().join("figs/gauß_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

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
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_merging_real_smoothed() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let data_cube = open_image(&root.join("full_cube_smoothed.fits"));
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_smoothed_merging_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}

#[cfg(feature = "plots")]
#[test]
fn test_segmenting_real_smoothed() {
  //Load image -> pick image with no NaN's
  println!("Loading reduced data cube");
  let root = get_root_path();
  let data_cube = open_image(&root.join("full_cube_smoothed.fits"));
  let img = data_cube.slice(nd::s![.., .., 120]);

  //make output folder and configure the watershed transform
  let root = root.join("figs/real_smoothed_segmenting_test/");
  if !root.exists() {
    std::fs::create_dir(&root).unwrap();
  }
  let watershed = TransformBuilder::new_merging(&root).build().unwrap();

  //run pre-processor and find minima
  let img = watershed.pre_processor(img.view());
  let mins = &watershed.find_local_minima(img.view());

  //Plot original
  rustronomy_watershed::plotting::plot_slice(img.view(), &root.join("original.png"), color_maps::viridis)
    .unwrap();

  //Do transform
  watershed.transform(img.view(), mins);
}
