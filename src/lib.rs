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

#![doc(
  html_logo_url = "https://raw.githubusercontent.com/smups/rustronomy/main/logos/Rustronomy_ferris.png?raw=true"
)]
//! Rustronomy-watershed is a pure-rust implementation of the segmenting and merging
//! watershed algorithms (see Digabel & Lantuéjoul, 1978[^1]).
//!
//! # Features
//! Two main versions of the watershed algorithm are included in this crate.
//! 1. The *merging* watershed algorithm, which is a void-filling algorithm that
//! can be used to identify connected regions in image.
//! 2. The *segmenting* watershed algorithm, which is a well-known image
//! segmentation algorithm.
//!
//! In addition, `rustronomy-watershed` provides extra functionality which can be
//! accessed via cargo feature gates. A list of all additional features [can be found
//! below](#cargo-feature-gates).
//!
//!
//! # Quickstart
//! To use the latest release of Rustronomy-watershed in a cargo project, add
//! the rustronomy-watershed crate as a dependency to your `Cargo.toml` file:
//! ```toml
//! [dependencies]
//! rustronomy-watershed = "0.2.0"
//! ```
//! To use Rustronomy-fits in a Jupyter notebook, execute a cell containing the
//! following code:
//! ```rust
//! :dep rustronomy-watershed = {version = "0.2"}
//! ```
//! If you want to use the latest (unstable) development version of
//! rustronomy-watershed, you can do so by using the `git` field (which fetches
//! the latest version from the repo) rather than the `version` field
//! (which downloads the latest released version from crates.io).
//! ```
//! {git = "https://github.com/smups/rustronomy-watershed"}
//! ```
//!
//! ## Short example: computing the Watershed transform of a random field
//! `rustronomy-watershed` uses the commonly used "builder pattern" to configure
//! the watershed transform before executing it. To configure a transform,
//! create an instance of the `TransformBuilder` struct. Once you are done specifying
//! options for the builder struct using its associated functions, call the
//! `build_merging()` or `build_segmenting()` functions to generate a
//! (`Sync`&`Send`) watershed transform struct, which you can now use to
//! execute the configured transform.
//!
//! In this example, we compute the watershed transform of a uniform random field.
//! The random field can be generated with the `ndarray_rand` crate. To configure a
//! new watershed transform, one can use the `TransformBuilder` struct which is
//! included in the `rustronomy_watershed` prelude.
//! ```rust
//! use ndarray as nd;
//! use rustronomy_watershed::prelude::*;
//! use ndarray_rand::{rand_distr::Uniform, RandomExt};
//!
//! //Create a random uniform distribution
//! let rf = nd::Array2::<u8>::random((512, 512), Uniform::new(0, 254));
//! //Set-up the watershed transform
//! let watershed = TransformBuilder::default().build_segmenting().unwrap();
//! //Find minima of the random field (to be used as seeds)
//! let rf_mins = watershed.find_local_minima(rf.view());
//! //Execute the watershed transform
//! let output = watershed.transform(rf.view(), &rf_mins);
//! ```
//! [^1]: H. Digabel and C. Lantuéjoul. **Iterative algorithms.** *In Actes du Second Symposium Européen d’Analyse Quantitative des Microstructures en Sciences des Matériaux, Biologie et Medécine*, October 1978.
//!
//! # Cargo feature gates
//! *By default, all features behind cargo feature gates are **disabled***
//! - `jemalloc`: this feature enables the [jemalloc allocator](https://jemalloc.net).
//! From the jemalloc website: *"jemalloc is a general purpose `malloc`(3)
//! implementation that emphasizes fragmentation avoidance and scalable concurrency
//! support."*. Jemalloc is enabled though usage of the `jemalloc` crate, which
//! increases compile times considerably. However, enabling this feature can also
//! greatly improve run-time performance, especially on machines with more (>6 or so)
//! cores. To compile `rustronomy-watershed` with the `jemalloc` feature,
//! jemalloc must be installed on the host system.
//! - `plots`: with this feature enabled, `rustronomy-watershed` will generate a
//! plot of the watershed-transform each time the water level is increased.
//! Plotting support adds the `plotters` crate as a dependency, which increases
//! compile times and requires the installation of some packages on linux
//! systems, [see the `plotters` documentation for details](https://docs.rs/plotters/).
//! - `progress`: this feature enables progress bars for the watershed algorithm.
//! Enabling this feature adds the `indicatif` crate as a dependency,
//! which should not considerably slow down compile times.
//! - `debug`: this feature enables debug and performance monitoring output. This
//! can negatively impact performance. Enabling this feature does not add additional
//! dependencies.
//!
//! ## `plots` feature gate
//! Enabling the `plots` feature gate adds two new methods to the `TransformBuilder`
//! struct: `set_plot_colour_map`, which can be used to set the colour map that
//! will be used by `plotters` to generate the images and `set_plot_folder`, which
//! can be used to specify folder where the generated images should be placed. If
//! no output folder is specified when the `plots` feature is enabled, no plots will
//! be generated (code will still compile).
//!
//! The generated plots are png files with no text. Each pixel in the generated
//! images corresponds 1:1 to a pixel in the input array.

//Unconditional imports
use ndarray as nd;
use num_traits::{Num, ToPrimitive};
use rand::{seq::SliceRandom, Rng};
use rayon::prelude::*;

//Set Jemalloc as the global allocator for this crate
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

//Progress bar (conditional)
#[cfg(feature = "progress")]
use indicatif;

//Constants for pixels that have to be left uncoloured, or have to be coloured
pub const UNCOLOURED: usize = 0;
pub const NORMAL_MAX: u8 = u8::MAX - 1;
pub const ALWAYS_FILL: u8 = u8::MIN;
pub const NEVER_FILL: u8 = u8::MAX;

//Utility prelude for batch import
pub mod prelude {
  pub use crate::{MergingWatershed, TransformBuilder, Watershed, WatershedUtils};
  #[cfg(feature = "plots")]
  pub mod color_maps {
    pub use crate::plotting::grey_scale;
    pub use crate::plotting::inferno;
    pub use crate::plotting::magma;
    pub use crate::plotting::plasma;
    pub use crate::plotting::viridis;
  }
}

////////////////////////////////////////////////////////////////////////////////
//                              HELPER FUNCTIONS                              //
////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "progress")]
fn set_up_bar(water_max: u8) -> indicatif::ProgressBar {
  const TEMPLATE: &str = "{spinner}[{elapsed}/{duration}] water level {pos}/{len}{bar:60}";
  let style = indicatif::ProgressStyle::with_template(TEMPLATE);
  let bar = indicatif::ProgressBar::new(water_max as u64);
  bar.set_style(style.unwrap());
  return bar;
}

#[inline]
fn neighbours_8con(index: &(usize, usize)) -> Vec<(usize, usize)> {
  let (x, y): (isize, isize) = (index.0 as isize, index.1 as isize);
  [
    (x + 1, y),
    (x + 1, y + 1),
    (x + 1, y - 1),
    (x, y + 1),
    (x, y - 1),
    (x - 1, y),
    (x - 1, y + 1),
    (x - 1, y - 1),
  ]
  .iter()
  .filter_map(|&(x, y)| if x < 0 || y < 0 { None } else { Some((x as usize, y as usize)) })
  .collect()
}

#[inline]
fn neighbours_4con(index: &(usize, usize)) -> Vec<(usize, usize)> {
  let (x, y): (isize, isize) = (index.0 as isize, index.1 as isize);
  [(x + 1, y), (x, y + 1), (x, y - 1), (x - 1, y)]
    .iter()
    .filter_map(|&(x, y)| if x < 0 || y < 0 { None } else { Some((x as usize, y as usize)) })
    .collect()
}

fn find_flooded_px(
  img: nd::ArrayView2<u8>,
  cols: nd::ArrayView2<usize>,
  lvl: u8,
) -> Vec<((usize, usize), usize)> {
  //Window size and index of center window pixel
  const WINDOW: (usize, usize) = (3, 3);
  const MID: (usize, usize) = (1, 1);

  /*
    We lock-step through (3x3) windows of both the input image and the output
    watershed (coloured water). We only consider the centre pixel, since all the
    windows overlap anyways. The index of the nd::Zip function is the (0,0) index
    of the window, so the index of the target pixel is at window_idx + (1,1).

    For each target pixel we:
      1. Check if it is flooded: YES -> continue, NO -> ignore px
      2. Check if it is uncoloured: YES -> continue, NO -> ignore px
      3. Check if at least one of the window pixels is coloured
        YES -> continue, NO -> ignore px
      4. Find the colours of the neighbouring pixels
        All same -> colour MID pixel with that colour
        Different -> pick a random colour
  */
  nd::Zip::indexed(cols.windows(WINDOW))
    .and(img.windows(WINDOW))
    .into_par_iter()
    //(1) Ignore unflooded pixels
    .filter(|&(_idx, _col_wd, img_wd)| img_wd[MID] <= lvl)
    //(2) Ignore already coloured pixels
    .filter(|&(_idx, col_wd, _img_wd)| col_wd[MID] == UNCOLOURED)
    //(3) Ignore pixels that do not border coloured pixels
    .filter(|&(_idx, col_wd, _img_wd)| {
      let neigh_idx_4c = neighbours_4con(&MID);
      !neigh_idx_4c.iter().all(|&idx| col_wd[idx] == UNCOLOURED)
    })
    //Set idx from upper left corner to target pixel, and ignore img window
    .map(|(idx, col_wd, _img_wd)| ((idx.0 + 1, idx.1 + 1), col_wd))
    //(4) Decide which colour our pixel should be
    .map(|(idx, col_wd)| {
      //Get indices of neighbouring pixels, then ask their colours
      let neigh_col_4c = neighbours_4con(&MID)
        .into_iter()
        .map(|neigh_idx| col_wd[neigh_idx])
        //Ignore uncoloured neighbours
        .filter(|&col| col != UNCOLOURED)
        .collect::<Vec<usize>>();

      //First neighbour will be our reference colour
      let col0 = *neigh_col_4c.get(0).expect("All neighbours were uncoloured!");
      if neigh_col_4c.iter().all(|&col| col == col0) {
        //All coloured neighbours have same colour
        (idx, col0)
      } else {
        //We have to pick a random colour
        let rand_idx = rand::thread_rng().gen_range(0..neigh_col_4c.len());
        let rand_col = *neigh_col_4c.get(rand_idx).expect("picking random px went wrong?");
        (idx, rand_col)
      }
    })
    .collect()
}

#[test]
fn test_find_px() {
  //This test assumes UNCOLOURED == 0, so it should fail
  assert!(UNCOLOURED == 0);
  let input = nd::array![
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 1, 0],
    [0, 0, 5, 4, 5, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 0],
  ];
  let colours = nd::array![
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
  ];
  let answer1 = [(1, 5), (2, 2), (4, 4), (5, 6)];
  let attempt1 = find_flooded_px(input.view(), colours.view(), 2)
    .into_iter()
    .map(|(x, _)| x)
    .collect::<Vec<_>>();
  for answer in answer1 {
    assert!(attempt1.contains(&answer))
  }
}

#[derive(Eq, Clone, Copy, Default, Debug)]
#[repr(transparent)]
struct Merge([usize; 2]);

// Merging 1 with 2 is the same as merging 2 with 1. So [1,2] == [2,1],
// as reflected by this impl
impl PartialEq for Merge {
  #[inline(always)]
  fn eq(&self, other: &Self) -> bool {
    let [x1, y1] = self.0;
    let [x2, y2] = other.0;
    (x1 == x2 && y1 == y2) || (x1 == y2 && y1 == x2)
  }
}

#[test]
fn test_merge_eq() {
  assert_eq!(Merge([1, 2]), Merge([2, 1]));
}

#[inline(always)]
fn sort_by_small_big(this: &Merge, that: &Merge) -> std::cmp::Ordering {
  use std::cmp::Ordering::*;
  if this == that {
    return Equal;
  }
  let (self_small, self_big) =
    if this.0[0] > this.0[1] { (this.0[0], this.0[1]) } else { (this.0[0], this.0[1]) };
  let (other_small, other_big) =
    if that.0[0] > that.0[1] { (that.0[0], that.0[1]) } else { (that.0[1], that.0[0]) };

  //First order on the basis of the smallest elements, then the largest ones
  if self_small < other_small {
    Less
  } else if self_small > other_small {
    Greater
  } else if self_big < other_big {
    Less
  } else {
    Greater
  }
}

#[test]
fn test_merge_ord_small_big() {
  use std::cmp::Ordering::*;
  let cmp = sort_by_small_big;
  assert_eq!(cmp(&Merge([2, 1]), &Merge([1, 1])), Greater);
  assert_eq!(cmp(&Merge([1, 1]), &Merge([1, 2])), Less);
  assert_eq!(cmp(&Merge([2, 1]), &Merge([1, 2])), Equal);
  assert_eq!(cmp(&Merge([3, 8]), &Merge([4, 5])), Less);
}

#[inline(always)]
fn sort_by_big_small(this: &Merge, that: &Merge) -> std::cmp::Ordering {
  use std::cmp::Ordering::*;
  if this == that {
    return Equal;
  }
  let (self_small, self_big) =
    if this.0[0] > this.0[1] { (this.0[0], this.0[1]) } else { (this.0[0], this.0[1]) };
  let (other_small, other_big) =
    if that.0[0] > that.0[1] { (that.0[0], that.0[1]) } else { (that.0[1], that.0[0]) };

  //First order on the basis of the smallest elements, then the largest ones
  if self_big < other_big {
    Less
  } else if self_big > other_big {
    Greater
  } else if self_small < other_small {
    Less
  } else {
    Greater
  }
}

#[test]
fn test_merge_ord_big_small() {
  use std::cmp::Ordering::*;
  let cmp = sort_by_big_small;
  assert_eq!(cmp(&Merge([2, 1]), &Merge([1, 1])), Greater);
  assert_eq!(cmp(&Merge([1, 1]), &Merge([1, 2])), Less);
  assert_eq!(cmp(&Merge([2, 1]), &Merge([1, 2])), Equal);
  assert_eq!(cmp(&Merge([3, 8]), &Merge([4, 5])), Greater);
}

impl From<[usize; 2]> for Merge {
  #[inline(always)]
  fn from(value: [usize; 2]) -> Self {
    Self(value)
  }
}

impl From<Merge> for [usize; 2] {
  #[inline(always)]
  fn from(value: Merge) -> Self {
    value.0
  }
}

fn find_merge(col: nd::ArrayView2<usize>) -> Vec<Merge> {
  //Window size and index of center window pixel
  const WINDOW: (usize, usize) = (3, 3);
  const MID: (usize, usize) = (1, 1);

  /*
    To find which regions to merge, we iterate in (3x3) windows over the current
    map of coloured pixels. We only consider the centre pixel, since all the
    windows overlap anyways.

    For each target pixel we:
      1. Check if the pixel is uncoloured. YES -> ignore, NO -> continue
      2. Check if the pixel has coloured neighbours. YES -> continue, NO -> ignore
      3. Check if the pixel has neighbours of different colours
        YES -> continue, NO -> ignore (this is a lake pixel)
      4. All neighbours that are left are now different colours than the MID px
        AND are not uncoloured. These pairs have to be merged
  */
  let mut merge = nd::Zip::from(col.windows(WINDOW))
    .into_par_iter()
    //(1) Check target pixel colour
    .filter(|&col_wd| col_wd.0[MID] != UNCOLOURED)
    //Map window to array of neighbour colours
    .map(|col_wd| -> (usize, Vec<usize>) {
      let own_col = col_wd.0[MID];
      let neighbour_cols = neighbours_4con(&MID)
        .into_iter()
        .map(|idx| col_wd.0[idx])
        .filter(|&col| col != UNCOLOURED)
        .collect();
      (own_col, neighbour_cols)
    })
    //(2) Ignore pixels with only uncoloured neighbours
    .filter(|(_own_col, neigh_col)| !neigh_col.is_empty())
    //(3) Collect neighbour colours. These have to be merged
    .map(|(own_col, neigh_col)| {
      neigh_col
        .into_iter()
        //(3a) Ignore mergers that merge a region with itself! ([1,1] and the likes)
        .filter_map(|c| if c == own_col { None } else { Some(Merge::from([own_col, c])) })
        .collect::<Vec<_>>()
    })
    .flatten()
    .collect::<Vec<_>>();

  //Remove duplicates (unstable sort may reorder duplicates, we don't care
  //because the whole point of sorting the vec is to get RID of duplicates!)
  merge.par_sort_unstable_by(sort_by_big_small);
  merge.dedup();
  merge.par_sort_unstable_by(sort_by_small_big);
  merge.dedup();
  return merge;
}

#[test]
fn test_find_merge() {
  //This test assumes UNCOLOURED == 0, so it should fail
  assert!(UNCOLOURED == 0);
  let input = nd::array![
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 2, 0, 1, 0],
    [0, 1, 1, 2, 2, 0, 1, 0],
    [0, 3, 3, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 0, 5, 5, 6, 0],
    [0, 4, 4, 0, 0, 5, 6, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
  ];
  let answer = vec![Merge([1, 2]), Merge([1, 3]), Merge([2, 3]), Merge([5, 6])];
  let result = find_merge(input.view());
  assert_eq!(answer.len(), result.len());
  assert!(result.iter().all(|x| answer.contains(x)));
}

fn make_colour_map(base_map: &mut [usize], pair_mergers: &[Merge]) {
  /* REDUCING 2-REGION MERGERS TO N-REGION MERGERS
    We are given a list of *locally* connected regions. For instance:
      (1,2) and (2,4)
    We have to turn locally connected regions into globally connected regions.
    In the example, the two locally connected regions are not connected directly,
    but only via region 2. They still have to merge into a single region:
      (1,2,3,4,5)
    There may be many steps between regions:
      (1,2) & (2,3) & (3,4) & (4,5)
    these should merge to:
      (1,2,3,4,5)
    regardless of the order in which they were specified!
  */
  let mut full_mergers: Vec<Vec<usize>> = Vec::new();

  'pair_loop: for &pair_merge in pair_mergers {
    //If pair_merge connects two full_merge regions, they have to be merged into
    //a single large region
    let [col1, col2]: [usize; 2] = pair_merge.into();
    let mut connect = [None, None];
    for (idx, region) in full_mergers.iter().enumerate() {
      if region.contains(&col1) && region.contains(&col2) {
        //This pair_merge was entirely contained within another region, it must
        //be a duplicate! We can skip to the next pair_merge
        continue 'pair_loop;
      } else if region.contains(&col1) || region.contains(&col2) {
        if connect[0].is_none() {
          connect[0] = Some(idx)
        } else if connect[1].is_none() {
          connect[1] = Some(idx);
          break;
        } else {
          panic!("Unreachable code path!")
        }
      }
    }

    if connect == [None, None] {
      //This pair_merge does not connect two full_merge regions, so it must be added
      //as its own full_merge region
      full_mergers.push(vec![col1, col2]);
    } else if let [Some(reg_idx), None] = connect {
      //This pair_merge *does* connect with another region, but only one.
      let reg = full_mergers.get_mut(reg_idx).unwrap();
      reg.extend_from_slice(&[col1, col2]);
      reg.sort();
      reg.dedup();
    } else if let [Some(reg_idx1), Some(reg_idx2)] = connect {
      //This pair_merge connects two regions, we must merge pair_merge AND both
      //regions at the same time.

      //Obtain a mutable ref to both regions (we'll drain one of them)
      //This code is messy thanks to the borrow checker
      let (reg1, reg2) = {
        let (larger, smaller) =
          if reg_idx1 > reg_idx2 { (reg_idx1, reg_idx2) } else { (reg_idx2, reg_idx1) };
        let (head, tail) = full_mergers.split_at_mut(smaller + 1);
        (&mut head[smaller], &mut tail[larger - smaller - 1])
      };

      //Drain region2 into region 1.
      //We do not have to append col1 or col2 because they are already contained
      //in reg1 and reg2. That is why we are merging them after all.
      reg1.append(reg2);
    }

    //remove empty regions
    full_mergers = full_mergers.into_iter().filter(|region| !region.is_empty()).collect();
  }

  for merge in full_mergers {
    let merged_col = *merge.get(0).expect("tried to merge zero regions");
    base_map.iter_mut().filter(|x| merge.contains(x)).for_each(|x| *x = merged_col);
  }
}

#[test]
fn test_make_colour_map() {
  //This test assumes UNCOLOURED == 0, so it should fail
  assert!(UNCOLOURED == 0);
  let mut cmap;
  let rng = &mut rand::thread_rng();
  for _ in 0..10 {
    //Test merging of once-connected region
    cmap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    make_colour_map(&mut cmap, &vec![Merge([1, 2])]);
    assert!(cmap == [0, 1, 1, 3, 4, 5, 6, 7, 8, 9]);

    //Now test multiple non-connected regions
    cmap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut input = vec![Merge([1, 2]), Merge([8, 9])];
    input.shuffle(rng);
    make_colour_map(&mut cmap, &input);
    assert!(cmap == [0, 1, 1, 3, 4, 5, 6, 7, 8, 8]);

    //Now test multiple *connected* regions
    cmap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut input = vec![Merge([1, 2]), Merge([2, 3])];
    input.shuffle(rng);
    make_colour_map(&mut cmap, &input);
    assert!(cmap == [0, 1, 1, 1, 4, 5, 6, 7, 8, 9]);

    //Two consecutive mergers
    cmap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut input = vec![Merge([1, 2]), Merge([8, 9])];
    input.shuffle(rng);
    make_colour_map(&mut cmap, &input);
    let mut input = vec![Merge([1, 7]), Merge([7, 8])];
    input.shuffle(rng);
    make_colour_map(&mut cmap, &input);
    assert!(cmap == [0, 1, 1, 3, 4, 5, 6, 1, 1, 1]);

    //Repeated merger (somehow)
    cmap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut input = vec![Merge([1, 2]), Merge([3, 2]), Merge([2, 1])];
    input.shuffle(rng);
    make_colour_map(&mut cmap, &input);
    assert!(cmap == [0, 1, 1, 1, 4, 5, 6, 7, 8, 9]);
  }
}

#[inline(always)]
fn recolour(mut canvas: nd::ArrayViewMut2<usize>, colour_map: &[usize]) {
  canvas.mapv_inplace(|px| colour_map[px])
}

#[test]
fn test_recolour() {
  //This test assumes UNCOLOURED == 0, so it should fail
  assert!(UNCOLOURED == 0);
  let mut input = nd::array![
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 2, 0, 1, 0],
    [0, 1, 1, 2, 2, 0, 1, 0],
    [0, 3, 3, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 0, 5, 5, 6, 0],
    [0, 4, 4, 0, 0, 5, 6, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
  ];
  let cmap = [0, 1, 1, 1, 4, 5, 5];
  let answer = nd::array![
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 4, 0, 5, 5, 5, 0],
    [0, 4, 4, 0, 0, 5, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
  ];
  recolour(input.view_mut(), &cmap);
  assert_eq!(answer, input);

  //Test that changing values no longer in the image does nothing
  let cmap = [0, 1, 13498683, 13458, 4, 5, 134707134];
  recolour(input.view_mut(), &cmap);
  assert_eq!(answer, input);
}

#[inline]
fn find_lake_sizes(ctx: HookCtx) -> (u8, Vec<usize>) {
  let mut lake_sizes = vec![0usize; ctx.colours.len() + 1];
  ctx.colours.iter().for_each(|&x| {
    *lake_sizes.get_mut(x).unwrap() += 1;
  });
  (ctx.water_level, lake_sizes)
}

////////////////////////////////////////////////////////////////////////////////
//                             OPTIONAL MODULES                               //
////////////////////////////////////////////////////////////////////////////////
#[cfg(feature = "debug")]
mod performance_monitoring {

  #[derive(Clone, Debug, Default)]
  pub struct PerfReport {
    pub big_iter_ms: Vec<usize>,
    pub colouring_mus: Vec<usize>,
    pub loops: usize,
    pub merge_ms: usize,
    pub lake_count_ms: usize,
    pub total_ms: usize,
  }

  impl PerfReport {
    pub fn iter_avg(&self) -> f64 {
      let num = self.big_iter_ms.len() as f64;
      self.big_iter_ms.iter().map(|&x| x as f64).sum::<f64>() / num
    }
    pub fn iter_total(&self) -> f64 {
      self.big_iter_ms.iter().map(|&x| x as f64).sum()
    }
    pub fn colour_avg(&self) -> f64 {
      let num = self.big_iter_ms.len() as f64;
      self.colouring_mus.iter().map(|&x| x as f64).sum::<f64>() / num
    }
    pub fn colour_total(&self) -> f64 {
      self.colouring_mus.iter().map(|&x| x as f64).sum()
    }
  }

  impl std::fmt::Display for PerfReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      writeln!(f, ">---------[Performance Summary]---------")?;
      writeln!(f, ">  Looped {}x", self.loops)?;
      writeln!(f, ">  Iteration Average: {:.1}ms; Σ {:.0}ms", self.iter_avg(), self.iter_total())?;
      writeln!(
        f,
        ">  Colouring Average: {:.1}µs; Σ {:.0}µs",
        self.colour_avg(),
        self.colour_total()
      )?;
      writeln!(f, ">  Merging: {}ms", self.merge_ms)?;
      writeln!(f, ">  Counting Lakes: {}ms", self.lake_count_ms)?;
      writeln!(f, ">--------------------------------+ total")?;
      writeln!(
        f,
        ">  {}ms with {:.1}ms overhead (Δt)",
        self.total_ms,
        self.total_ms as f64
          - self.iter_total()
          - self.colour_total() / 1000.0
          - self.merge_ms as f64
          - self.lake_count_ms as f64
      )
    }
  }
}

#[cfg(feature = "plots")]
/// This module contains all the code required to generate images from the
/// watershed array, including all the included colour maps.
pub mod plotting {
  use ndarray as nd;
  use num_traits::ToPrimitive;
  use plotters::prelude::*;
  use std::{error::Error, path::Path};

  //Colour for nan px
  const NAN_COL: RGBColor = BLACK;

  //Module that contains hardcoded colourmaps from matplotlib
  mod color_maps;

  pub fn plot_slice<'a, T>(
    slice: nd::ArrayView2<'a, T>,
    file_name: &Path,
    color_map: fn(count: T, min: T, max: T) -> Result<RGBColor, Box<dyn Error>>,
  ) -> Result<(), Box<dyn Error>>
  where
    T: Default + std::fmt::Display + std::cmp::PartialOrd + ToPrimitive + Copy,
  {
    //Get min and max vals of slice
    let min = slice.iter().fold(T::default(), |f: T, x: &T| if *x < f { *x } else { f });
    let max = slice.iter().fold(T::default(), |f: T, x: &T| if *x > f { *x } else { f });

    //Get the size of the slice
    let x_size = slice.shape()[0] as u32;
    let y_size = slice.shape()[1] as u32;

    //Make new fig
    let root = BitMapBackend::new(file_name, (x_size, y_size)).into_drawing_area();
    root.fill(&WHITE)?;

    //make empty drawing area in fig
    let mut chart = ChartBuilder::on(&root).build_cartesian_2d(0..x_size, 0..y_size)?;
    chart.configure_mesh().disable_mesh().disable_axes().draw()?;
    let plotting_area = chart.plotting_area();

    //fill pixels
    for ((x, y), px) in slice.indexed_iter() {
      plotting_area.draw_pixel((x as u32, y as u32), &color_map(*px, min, max)?)?
    }

    //save file
    root.present()?;

    #[cfg(feature = "debug")]
    println!("slice saved as png: {file_name:?}; max:{max:2}, min:{min:2}");
    Ok(())
  }

  #[inline(always)]
  pub fn grey_scale<T>(count: T, min: T, max: T) -> Result<RGBColor, Box<dyn Error>>
  where
    T: std::fmt::Display + std::cmp::PartialOrd + ToPrimitive,
  {
    if count <= min {
      //This is a NAN pixel, fill it with the NaN colour
      Ok(NAN_COL)
    } else {
      //Grayscale value
      let gray = ((255.0f64 * count.to_f64().unwrap() + min.to_f64().unwrap())
        / max.to_f64().unwrap()) as u8;
      Ok(RGBColor(gray, gray, gray))
    }
  }

  #[inline(always)]
  pub fn viridis<T>(count: T, min: T, max: T) -> Result<RGBColor, Box<dyn Error>>
  where
    T: std::fmt::Display + std::cmp::PartialOrd + ToPrimitive,
  {
    if count <= min {
      //This is a NAN pixel, fill it with the NaN colour
      Ok(NAN_COL)
    } else {
      //Grayscale value
      let gray = ((255.0f64 * count.to_f64().unwrap() + min.to_f64().unwrap())
        / max.to_f64().unwrap()) as usize;
      let color = color_maps::VIRIDIS[gray];
      Ok(RGBColor((color[0] * 256.0) as u8, (color[1] * 256.0) as u8, (color[2] * 256.0) as u8))
    }
  }

  #[inline(always)]
  pub fn magma<T>(count: T, min: T, max: T) -> Result<RGBColor, Box<dyn Error>>
  where
    T: std::fmt::Display + std::cmp::PartialOrd + ToPrimitive,
  {
    if count <= min {
      //This is a NAN pixel, fill it with the NaN colour
      Ok(NAN_COL)
    } else {
      //Grayscale value
      let gray = ((255.0f64 * count.to_f64().unwrap() + min.to_f64().unwrap())
        / max.to_f64().unwrap()) as usize;
      let color = color_maps::MAGMA[gray];
      Ok(RGBColor((color[0] * 256.0) as u8, (color[1] * 256.0) as u8, (color[2] * 256.0) as u8))
    }
  }

  #[inline(always)]
  pub fn plasma<T>(count: T, min: T, max: T) -> Result<RGBColor, Box<dyn Error>>
  where
    T: std::fmt::Display + std::cmp::PartialOrd + ToPrimitive,
  {
    if count <= min {
      //This is a NAN pixel, fill it with the NaN colour
      Ok(NAN_COL)
    } else {
      //Grayscale value
      let gray = ((255.0f64 * count.to_f64().unwrap() + min.to_f64().unwrap())
        / max.to_f64().unwrap()) as usize;
      let color = color_maps::PLASMA[gray];
      Ok(RGBColor((color[0] * 256.0) as u8, (color[1] * 256.0) as u8, (color[2] * 256.0) as u8))
    }
  }

  #[inline(always)]
  pub fn inferno<T>(count: T, min: T, max: T) -> Result<RGBColor, Box<dyn Error>>
  where
    T: std::fmt::Display + std::cmp::PartialOrd + ToPrimitive,
  {
    if count <= min {
      //This is a NAN pixel, fill it with the NaN colour
      Ok(NAN_COL)
    } else {
      //Grayscale value
      let gray = ((255.0f64 * count.to_f64().unwrap() + min.to_f64().unwrap())
        / max.to_f64().unwrap()) as usize;
      let color = color_maps::INFERNO[gray];
      Ok(RGBColor((color[0] * 256.0) as u8, (color[1] * 256.0) as u8, (color[2] * 256.0) as u8))
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//                          WATERSHED TRANSFORMS                              //
////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "plots")]
use plotters::prelude::*;

#[derive(Clone)]
pub struct HookCtx<'a> {
  pub water_level: u8,
  pub max_water_level: u8,
  pub image: nd::ArrayView2<'a, u8>,
  pub colours: nd::ArrayView2<'a, usize>,
  pub seeds: &'a [(usize, (usize, usize))],
}

impl<'a> HookCtx<'a> {
  fn ctx(
    water_level: u8,
    max_water_level: u8,
    image: nd::ArrayView2<'a, u8>,
    colours: nd::ArrayView2<'a, usize>,
    seeds: &'a [(usize, (usize, usize))],
  ) -> Self {
    HookCtx { water_level, max_water_level, image, colours, seeds }
  }
}

#[derive(Clone)]
/// Builder for configuring a watershed transform.
///
/// Use the `new_segmenting()` associated function to start configuring a
/// segmenting watershed transform. Use the `default()` method to start configuring
/// a watershed transform. Once you have enabled the desired functionality,
/// a watershed transform struct can be generated with the `build_segmenting()`
/// and `build_merging()` associated functions. These return a struct of the type
/// `SegmentingWatershed` and `MergingWatershed` respectively, which can be
/// shared between threads.
/// 
/// ## `default()` vs `new()` and custom hooks
/// tl;dr: Use `new()` if you want to run a custom hook each time a the water
/// level is raised during the watershed transform. Otherwise, use `default()`.
/// 
/// A `TransformBuilder` struct can be obtained with both the `default()` and
/// `new()` functions implemented for both. The main difference between the two
/// is the resulting type:
/// - `default()` results in a `TransformBuilder<()>`
/// - `new()` results in a `TransformBuilder<T>` 
/// The default type for `T` is `()`. `T` is only used when specifying a custom
/// function run by the watershed transform each time the water level is raised.
/// This "hook" has the type `fn(HookCtx) -> T`.
/// 
/// Sadly, Rust is not able to determine that `T` should take the default `()`
/// type if no hook is configured during the builder phase (which can be done with
/// the `set_wlvl_hook` function). Therefore, `Default` is only implemented for
/// `TransformBuilder<()>` (so not for all `T`) and can be used to circumvent this
/// limitation of the type inference engine.
///
/// ## `plots` feature
/// Enabling the `plots` feature gate adds two new methods to the `TransformBuilder`
/// struct: `set_plot_colour_map`, which can be used to set the colour map that
/// will be used by `plotters` to generate the images and `set_plot_folder`, which
/// can be used to specify folder where the generated images should be placed. If
/// no output folder is specified when the `plots` feature is enabled, no plots will
/// be generated (code will still compile).
///
/// ## `enable_edge_correction`
/// Calling the `enable_edge_correction` method on the builder signals the
/// watershed implementations that they should make sure that the edges of the
/// image are properly included in the watershed transform. This option is disabled
/// by default since performing this "edge correction" can incur a significant
/// performance/memory usage hit.
pub struct TransformBuilder<T = ()> {
  //Plotting options
  #[cfg(feature = "plots")]
  plot_path: Option<std::path::PathBuf>,
  #[cfg(feature = "plots")]
  plot_colour_map: Option<
    fn(count: usize, min: usize, max: usize) -> Result<RGBColor, Box<dyn std::error::Error>>,
  >,

  //Basic transform options
  max_water_level: u8,
  edge_correction: bool,

  //Hooks
  wlvl_hook: Option<fn(HookCtx) -> T>,
}

impl Default for TransformBuilder<()> {
  fn default() -> Self {
    TransformBuilder::new()
  }
}

impl<T> TransformBuilder<T> {
  /// Creates a new instance of `TransformBuilder<T>` which can be used to
  /// construct a watershed transform that runs custom code each time the water
  /// level is raised. If you do not use this functionality, use `default()`
  /// instead.
  pub const fn new() -> Self {
    TransformBuilder {
      #[cfg(feature = "plots")]
      plot_path: None,
      #[cfg(feature = "plots")]
      plot_colour_map: None,
      max_water_level: NORMAL_MAX,
      edge_correction: false,
      wlvl_hook: None,
    }
  }

  /// Set the maximum water level that the transform will reach. Note that the
  /// maximum water level may not be set higher than `u8::MAX - 1` (254).
  pub const fn set_max_water_lvl(mut self, max_water_lvl: u8) -> Self {
    self.max_water_level = max_water_lvl;
    self
  }

  /// Enables edge correction. Turning this setting on properly colours the edges
  /// of the input image at the cost of increased memory consumption and two
  /// full-image copies.
  pub const fn enable_edge_correction(mut self) -> Self {
    self.edge_correction = true;
    self
  }

  /// Sets the water level hook. This function pointer is called every time the
  /// water level is raised and may be used to implement custom statistics.
  /// Implementations of the watershed algorithm that do no visit all water levels
  /// are not guaranteed to call this hook at all.
  pub const fn set_wlvl_hook(mut self, hook: fn(HookCtx) -> T) -> Self {
    self.wlvl_hook = Some(hook);
    self
  }

  #[cfg(feature = "plots")]
  /// Set a custom colour map to be used by `plotters` when generating images
  /// of the watershed transform.
  pub const fn set_plot_colour_map(
    mut self,
    colour_map: fn(
      count: usize,
      min: usize,
      max: usize,
    ) -> Result<RGBColor, Box<dyn std::error::Error>>,
  ) -> Self {
    self.plot_colour_map = Some(colour_map);
    self
  }

  #[cfg(feature = "plots")]
  /// Set output folder for the images generated during the watershed transform.
  /// If no output folder is specified, no images will be generated, even with
  /// the `plots` feature gate enabled.
  pub fn set_plot_folder(mut self, path: &std::path::Path) -> Self {
    self.plot_path = Some(path.to_path_buf());
    self
  }

  /// Build a `MergingWatershed<T>` from the current builder
  /// configuration.
  pub fn build_merging(self) -> Result<MergingWatershed<T>, BuildErr> {
    //Check if the max water level makes sense
    if self.max_water_level > NORMAL_MAX {
      Err(BuildErr::MaxToHigh(self.max_water_level))?
    } else if self.max_water_level <= ALWAYS_FILL {
      Err(BuildErr::MaxToLow(self.max_water_level))?
    }

    Ok(MergingWatershed {
      //Plot options
      #[cfg(feature = "plots")]
      plot_path: self.plot_path,
      #[cfg(feature = "plots")]
      plot_colour_map: self.plot_colour_map.unwrap_or(plotting::viridis),

      //Required options
      max_water_level: self.max_water_level,
      edge_correction: self.edge_correction,

      //Hooks
      wlvl_hook: self.wlvl_hook,
    })
  }

  /// Build a `SegmentingWatershed<T>` from the current builder
  /// configuration.
  pub fn build_segmenting(self) -> Result<SegmentingWatershed<T>, BuildErr> {
    //Check if the max water level makes sense
    if self.max_water_level > NORMAL_MAX {
      Err(BuildErr::MaxToHigh(self.max_water_level))?
    } else if self.max_water_level <= ALWAYS_FILL {
      Err(BuildErr::MaxToLow(self.max_water_level))?
    }

    Ok(SegmentingWatershed {
      //Plot options
      #[cfg(feature = "plots")]
      plot_path: self.plot_path,
      #[cfg(feature = "plots")]
      plot_colour_map: self.plot_colour_map.unwrap_or(plotting::viridis),

      //Required options
      max_water_level: self.max_water_level,
      edge_correction: self.edge_correction,

      //Hooks
      wlvl_hook: self.wlvl_hook,
    })
  }
}

#[derive(Debug, Clone)]
/// Errors that may occur during the build process
pub enum BuildErr {
  MaxToHigh(u8),
  MaxToLow(u8)
}

impl std::error::Error for BuildErr {}
impl std::fmt::Display for BuildErr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use BuildErr::*;
    match self {
      MaxToHigh(max) => write!(f, "Maximum water level set to {max}, which is higher than the maximum allowed value {NORMAL_MAX}"),
      MaxToLow(max) => write!(f, "Maximum water level set to {max}, which is lower than the minimum allowed value {NEVER_FILL}")
    }
  }
}

/// This trait contains useful functions for preparing images to be used as input
/// for a watershed transform
pub trait WatershedUtils {
  /// The `pre_processor` function can convert an array of any numeric data-type
  /// `T` into an array of `u8`. It converts special float values (if `T` is a
  /// float type) to `u8` values that implementations of the watershed transform
  /// in this crate know how to handle.
  ///
  /// In particular: `NaN` and positive infinity are mapped to the special
  /// `NEVER_FILL` value, and negative infinity is mapped to the special `ALWAYS_FILL`
  /// value.
  ///
  /// This function also automatically clamps the pixel values of the array to
  /// the full range of non-special `u8` values.
  fn pre_processor<T, D>(&self, img: nd::ArrayView<T, D>) -> nd::Array<u8, D>
  where
    T: Num + Copy + ToPrimitive + PartialOrd,
    D: nd::Dimension,
  {
    self.pre_processor_with_max::<NORMAL_MAX, T, D>(img)
  }

  // The `pre_processor_with` function can convert an array of any numeric data-type
  /// `T` into an array of `u8`. It converts special float values (if `T` is a
  /// float type) to `u8` values that implementations of the watershed transform
  /// in this crate know how to handle.
  ///
  /// In particular: `NaN` and positive infinity are mapped to the special
  /// `NEVER_FILL` value, and negative infinity is mapped to the special `ALWAYS_FILL`
  /// value.
  ///
  /// This function also automatically clamps the pixel values of the array to
  /// the range between 0 and the supplied MAX constant.
  ///
  /// # example usage
  /// This function can be used as a replacement for the pre-processor if you want
  /// to normalise the input to the water transform to a different range than
  /// `1..u8::MAX-1`, like so:
  /// ```rust
  /// use rustronomy_watershed::prelude::*;
  /// use ndarray as nd;
  /// use ndarray_rand::{rand_distr::Uniform, RandomExt};
  ///
  /// //Set custom maximum waterlevel
  /// const MY_MAX: u8 = 127;
  ///
  /// //Create a random uniform distribution
  /// let rf = nd::Array2::<f64>::random((512, 512), Uniform::new(0.0, 1.0));
  ///
  /// //Set-up the watershed transform
  /// let watershed = TransformBuilder::default()
  ///     .set_max_water_lvl(MY_MAX)
  ///     .build_segmenting()
  ///     .unwrap();
  ///
  /// //Run pre-processor (using turbofish syntax)
  /// let rf = watershed.pre_processor_with_max::<{MY_MAX}, _, _>(rf.view());
  ///
  /// //Find minima of the random field (to be used as seeds)
  /// let rf_mins = watershed.find_local_minima(rf.view());
  /// //Execute the watershed transform
  /// let output = watershed.transform(rf.view(), &rf_mins);
  /// ```
  ///
  /// # panics
  /// This function panics if MAX is bigger than or equal to `NEVER_FILL`, or
  /// smaller than or equal to `ALWAYS_FILL`.
  fn pre_processor_with_max<const MAX: u8, T, D>(
    &self,
    img: nd::ArrayView<T, D>,
  ) -> nd::Array<u8, D>
  where
    T: Num + Copy + ToPrimitive + PartialOrd,
    D: nd::Dimension,
  {
    //Panic if MAX is invalid (dear compiler, please remove this code path ty <3)
    assert!(MAX < NEVER_FILL);
    assert!(MAX > ALWAYS_FILL);

    //Calculate max and min values
    let min = img
      .iter()
      .fold(T::zero(), |acc, x| if *x < acc && x.to_f64().unwrap().is_finite() { *x } else { acc })
      .to_f64()
      .unwrap();
    let max = img
      .iter()
      .fold(T::zero(), |acc, x| if *x > acc && x.to_f64().unwrap().is_finite() { *x } else { acc })
      .to_f64()
      .unwrap();

    //Map image to u8 range, taking care of NaN and infty
    img.mapv(|x| -> u8 {
      let float = x.to_f64().unwrap();
      if float.is_normal() {
        //Clamp value to [0,1] range and then to [0, u8::MAX)
        let normal = (float - min) / (max - min);
        (normal * MAX as f64).to_u8().unwrap()
      } else if float.is_infinite() && !float.is_sign_negative() {
        //negative infinity, always fill
        ALWAYS_FILL
      } else {
        //Nans and positive infinity, never fill
        NEVER_FILL
      }
    })
  }

  /// returns a vec of the positions of all the pixels that have a lower value
  /// than all their 8-way connected neighbours. Useful for generating seeds for
  /// the watershed transform.
  fn find_local_minima(&self, img: nd::ArrayView2<u8>) -> Vec<(usize, usize)> {
    //Window size and index of center window pixel
    const WINDOW: (usize, usize) = (3, 3);
    const MID: (usize, usize) = (1, 1);

    nd::Zip::indexed(img.windows(WINDOW))
      .into_par_iter()
      .filter_map(|(idx, window)| {
        //Yield only pixels that are lower than their surroundings
        let target_val = window[MID];
        let neighbour_vals: Vec<u8> =
          neighbours_8con(&MID).into_iter().map(|idx| window[idx]).collect();
        if neighbour_vals.into_iter().all(|val| val < target_val) {
          Some((idx.0 + 1, idx.1 + 1))
        } else {
          None
        }
      })
      .collect()
  }
}

impl<T> WatershedUtils for MergingWatershed<T> {}
impl<T> WatershedUtils for SegmentingWatershed<T> {}

/// Actual trait for performing the watershed transform. It is implemented in
/// different ways by different versions of the algorithm. This trait is dyn-safe,
/// which means that trait objects may be constructed from it.
pub trait Watershed<T = ()> {
  /// Returns watershed transform of input image.
  fn transform(&self, input: nd::ArrayView2<u8>, seeds: &[(usize, usize)]) -> nd::Array2<usize>;

  /// Runs the watershed transform, executing the hook specified by the
  /// `TransformBuilder` (if there is one) each time the water level is raised.
  /// The results from running the hook each time are collected into a vec and
  /// returned by this function.
  fn transform_with_hook(&self, input: nd::ArrayView2<u8>, seeds: &[(usize, usize)]) -> Vec<T>;

  /// Returns a Vec containing the areas of all the lakes per water level. The
  /// length of the nested Vec is always equal to the number of seeds, although
  /// some lakes may have zero area (especially for the merging transform,
  /// see docs for `MergingWatershed`)
  fn transform_to_list(
    &self,
    input: nd::ArrayView2<u8>,
    seeds: &[(usize, usize)],
  ) -> Vec<(u8, Vec<usize>)>;

  /// Returns a list of images where each image corresponds to a snapshot of the
  /// watershed transform at a particular water level.
  ///
  /// **Caution**: this function has to allocate a full image each time the water
  /// level is raised. This significantly increases memory usage. If you just
  /// want plots of the intermediate images, consider turning on the `plots`
  /// feature instead.
  fn transform_history(
    &self,
    input: nd::ArrayView2<u8>,
    seeds: &[(usize, usize)],
  ) -> Vec<(u8, nd::Array2<usize>)>;
}

/// Implementation of the merging watershed algorithm.
///
/// See crate-level documentation for a general introduction to the algorithm.
///
/// The merging watershed transform is a slight variation on the segmenting
/// algorithm (see docs of the `SegmentingWatershed` struct). Instead of creating
/// a wall whenever two lakes meet, the merging watershed transform merges the
/// two lakes.
///
/// On a statistics level, the main difference between the merging and segmenting
/// watershed transforms is that the number of distinct lakes in the merging
/// watershed transform depends on the features in the image rather than the
/// number of (somewhat arbitrarily chosen) lake-seeds. Therefore, one can do
/// statistics with the number of lakes. In addition, the output of the merging
/// transform does not depend strongly on the precise way the minima were chosen.
///
/// # Memory usage
/// The watershed transform creates an `Array2<usize>` of the same size as the
/// input array, which takes up a considerable amount of memory. In addition, it
/// allocates space for a colour-map (`Vec<usize>` with a length equal to the
/// number of seeds) and some other intermediate, smaller vec's. One can count on
/// the memory usage being about ~2.5x the size of the input array.
///
/// ## Memory usage of `transform_history`
/// The `transform_history` method makes a copy of the `Array2<usize>` image used
/// during the watershed transform to keep track of the intermediate images. As
/// such, it allocates a new `Array2<usize>` for each water level, which increases
/// memory usage by a factor equal to the maximum water level as configured by
/// the `TransformBuilder`.
///
/// # Output
/// The three methods of the `Watershed` trait each return a different set of
/// parameters based on the watershed transform of the input image:
/// - `transform` simply returns the watershed transform of the image. This is
/// not very interesting in the case of the merging watershed transform, since
/// its output is just an image with all pixels having the same colour.
/// - `transform_to_list` returns a list of the areas of all the lakes at each
/// water level. Its return type is`Vec<(u8, Vec<usize>)>`, where the `u8` equals
/// the water level and the `Vec<usize>` is the list of areas of all the lakes
/// at each water level. The `Vec<usize>` is the same length for each water level,
/// but may contain zero-sided lakes. The water levels are returned in order.
/// - `transform_history` returns a list of intermediate images of the watershed
/// transform at every water level. Its return type is `Vec<(u8, ndarray::Array2<usize>)`,
/// where the `u8` holds the water level that each `Array2` snapshot was taken at.
/// The water levels are returned in order.
///
/// # Artifacts and peculiarities
/// Due to some implementation details, the 1px-wide edges of the input array are
/// not accessible to the watershed transform. They will thus remain unfilled for
/// the entire duration of the transform.
///
/// A workaround can be enabled by calling `enable_edge_correction` on the
/// `TransformBuilder`. Enabling this setting copies the input image into a
/// new array, 1px wider on all sides. This padded array is then used as the
/// actual input to the watershed transform. The final output of the transform is
/// a copy of this intermediate array with the padding removed. The padding also
/// does not show up in the output of intermediate plots.
pub struct MergingWatershed<T = ()> {
  //Plot options
  #[cfg(feature = "plots")]
  plot_path: Option<std::path::PathBuf>,
  #[cfg(feature = "plots")]
  plot_colour_map:
    fn(count: usize, min: usize, max: usize) -> Result<RGBColor, Box<dyn std::error::Error>>,
  
  //Required options
  max_water_level: u8,
  edge_correction: bool,

  //Hooks
  wlvl_hook: Option<fn(HookCtx) -> T>,
}

impl<T> MergingWatershed<T> {
  fn clone_with_hook<U>(&self, hook: fn(HookCtx) -> U) -> MergingWatershed<U> {
    MergingWatershed {
      #[cfg(feature = "plots")]
      plot_path: self.plot_path.clone(),
      #[cfg(feature = "plots")]
      plot_colour_map: self.plot_colour_map,
      max_water_level: self.max_water_level,
      edge_correction: self.edge_correction,
      wlvl_hook: Some(hook),
    }
  }
}

impl<T> Watershed<T> for MergingWatershed<T> {
  fn transform_with_hook(&self, input: nd::ArrayView2<u8>, seeds: &[(usize, usize)]) -> Vec<T> {
    //(1a) make an image for holding the diffserent water colours
    let shape = if self.edge_correction {
      //If the edge correction is enabled, we have to pad the input with a 1px
      //wide border, which increases the size shape of the output image by two
      [input.shape()[0] + 2, input.shape()[1] + 2]
    } else {
      [input.shape()[0], input.shape()[1]]
    };
    let mut output = nd::Array2::<usize>::zeros(shape);

    //(1b) reshape the input image if necessary
    let mut padded_input =
      if self.edge_correction { Some(nd::Array2::<u8>::zeros(shape)) } else { None };
    let input = if self.edge_correction {
      //Copy the input pixel values into the new padded image
      nd::Zip::from(
        padded_input
          .as_mut()
          .expect("corrected_input was None, which should be impossible. Please report this bug.")
          .slice_mut(nd::s![1..(shape[0] - 1), 1..(shape[1] - 1)]),
      )
      .and(input)
      .into_par_iter()
      .for_each(|(a, &b)| *a = b);
      padded_input.as_ref().unwrap().view()
    } else {
      input.reborrow()
    };

    //(2) set "colours" for each of the starting points
    // The colours should range from 1 to seeds.len()
    let mut colours: Vec<usize> = (1..=seeds.len()).into_iter().collect();
    let seed_colours: Vec<_> =
      colours.iter().zip(seeds.iter()).map(|(col, (x, z))| (*col, (*x, *z))).collect();

    //Colour the starting pixels
    for (&idx, &col) in seeds.iter().zip(colours.iter()) {
      output[idx] = col;
    }
    //Set the zeroth colour to UNCOLOURED!
    colours.insert(UNCOLOURED, UNCOLOURED);

    #[cfg(feature = "debug")]
    println!("starting with {} lakes", colours.len());

    //(3) set-up progress bar
    #[cfg(feature = "progress")]
    let bar = set_up_bar(self.max_water_level);

    //(4) count lakes for all water levels
    (0..=self.max_water_level)
      .into_iter()
      .map(|water_level| {
        //(logging) make a new perfreport
        #[cfg(feature = "debug")]
        let mut perf = crate::performance_monitoring::PerfReport::default();
        #[cfg(feature = "debug")]
        let loop_start = std::time::Instant::now();

        /*(i) Colour all flooded pixels connected to a source
          We have to loop multiple times because there may be plateau's. These
          require us to colour more than just one neighbouring pixel -> we need
          to loop until there are no more uncoloured, flooded pixels connected to
          a source left.
        */
        'colouring_loop: loop {
          #[cfg(feature = "progress")]
          {
            bar.tick(); //Tick the progressbar
          }
          #[cfg(feature = "debug")]
          {
            perf.loops += 1;
          }

          #[cfg(feature = "debug")]
          let iter_start = std::time::Instant::now();

          /*(A) Find pixels to colour this iteration
            We first look for all pixels that are uncoloured, flooded and directly
            attached to a coloured pixel. We do this in parallel. We cannot, however,
            change the pixel colours *and* look for pixels to colour at the same time.
            That is why we collect all pixels to colour in a vector, and later update
            the map.
          */
          let pix_to_colour = find_flooded_px(input.view(), output.view(), water_level);

          #[cfg(feature = "debug")]
          perf.big_iter_ms.push(iter_start.elapsed().as_millis() as usize);

          /*(B) Colour pixels that we found in step (A)
            If there are no pixels to be coloured anymore, we can break from this
            loop and raise the water level
          */
          if pix_to_colour.is_empty() {
            //No more connected, flooded pixels left -> raise water level
            break 'colouring_loop;
          } else {
            //We have pixels to colour
            #[cfg(feature = "debug")]
            let colour_start = std::time::Instant::now();

            pix_to_colour.into_iter().for_each(|(idx, col)| {
              output[idx] = col;
            });

            #[cfg(feature = "debug")]
            perf.colouring_mus.push(colour_start.elapsed().as_micros() as usize);
          }
        }

        /* (ii) Merge all touching regions
          Now that we have coloured all colourable pixels, we have to start
          merging regions of different colours that border each other
          We do this by making a look-up table for the colours. Each colour can
          look-up what its new colour will be.
        */
        #[cfg(feature = "debug")]
        let merge_start = std::time::Instant::now();

        //(A) Find all colours that have to be merged
        let to_merge = find_merge(output.view());
        let num_mergers = to_merge.len();

        /*(B) construct a colour map
          The colour map holds the output colour at the index equal to the input
          colour. A 1->1 identity map is therefore just a vec with its index as an
          entry.

          The UNCOLOURED (0) colour always has to be mapped to UNCOLOURED!
        */
        make_colour_map(&mut colours, &to_merge);
        assert!(colours[UNCOLOURED] == UNCOLOURED);

        //(C) Recolour the canvas with the colour map if the map is not empty
        if num_mergers > 0 {
          recolour(output.view_mut(), &colours);
        }
        #[cfg(feature = "debug")]
        {
          perf.merge_ms = merge_start.elapsed().as_millis() as usize;
        }

        //(iii) Plot current state of the watershed transform
        #[cfg(feature = "plots")]
        if let Some(ref path) = self.plot_path {
          if let Err(err) = plotting::plot_slice(
            if self.edge_correction {
              //Do not plot the edge correction padding
              output.slice(nd::s![1..(shape[0] - 1), 1..(shape[1] - 1)])
            } else {
              output.view()
            },
            &path.join(&format!("ws_lvl{water_level}.png")),
            self.plot_colour_map,
          ) {
            println!("Could not make watershed plot. Error: {err}")
          }
        }

        //(iv) print performance report
        #[cfg(all(feature = "debug", feature = "progress"))]
        {
          //In this combination we have a progress bar, we should use it to print
          perf.total_ms = loop_start.elapsed().as_millis() as usize;
          bar.println(format!("{perf}"));
        }
        #[cfg(all(feature = "debug", not(feature = "progress")))]
        {
          //We do not have a progress bar, so a plain println! will have to do
          perf.total_ms = loop_start.elapsed().as_millis() as usize;
          println!("{perf}");
        }

        //(v) Update progressbar and plot stuff
        #[cfg(feature = "progress")]
        {
          bar.inc(1);
        }

        //(vi) Execute hook (if one is provided)
        self.wlvl_hook.and_then(|hook| {
          Some(hook(HookCtx::ctx(
            water_level,
            self.max_water_level,
            input.view(),
            output.view(),
            &seed_colours,
          )))
        })
      })
      .filter_map(|x| x)
      .collect()
  }

  fn transform(&self, input: nd::ArrayView2<u8>, _seeds: &[(usize, usize)]) -> nd::Array2<usize> {
    //Note: the implementation of `transform` is trivial for the merging transfo

    //(1) make an image for holding the different water colours
    let shape = [input.shape()[0], input.shape()[1]];
    let mut output = nd::Array2::<usize>::zeros(shape);

    //(2) give all pixels except the edge a different colour
    output.slice_mut(nd::s![1..shape[0] - 1, 1..shape[1] - 1]).mapv_inplace(|_| 123);

    //Return the transformed image
    return output;
  }

  fn transform_history(
    &self,
    input: nd::ArrayView2<u8>,
    seeds: &[(usize, usize)],
  ) -> Vec<(u8, nd::Array2<usize>)> {
    //(1) Make a copy of self with the appropriate hook
    let proper_transform =
      self.clone_with_hook(|ctx| (ctx.water_level, ctx.colours.to_owned()));

    //(2) Perform transform with new hook
    proper_transform.transform_with_hook(input, seeds)
  }

  fn transform_to_list(
    &self,
    input: nd::ArrayView2<u8>,
    seeds: &[(usize, usize)],
  ) -> Vec<(u8, Vec<usize>)> {
    //(1) Make a copy of self with the appropriate hook
    let proper_transform = self.clone_with_hook(find_lake_sizes);

    //(2) Perform transform with new hook
    proper_transform.transform_with_hook(input, seeds)
  }
}

/// Implementation of the segmenting watershed algorithm.
///
/// See crate-level documentation for a general introduction to the algorithm.
///
/// The segmenting watershed algorithm forms lakes from pre-defined local minima
/// by raising an imaginary water level. Once the water level increases past the
/// height of a minimum, it starts filling neighbouring pixels that are also
/// below the water level. These poodles grow larger as the water level rises.
///
/// When two lakes originating from different local minima meet, an infinitely
/// high wall separating the two is created. This is wall-building is what makes
/// this version of the watershed algorithm an image *segmentation* algorithm.
///
/// ## Memory usage of `transform_history`
/// The `transform_history` method makes a copy of the `Array2<usize>` image used
/// during the watershed transform to keep track of the intermediate images. As
/// such, it allocates a new `Array2<usize>` for each water level, which increases
/// memory usage by a factor equal to the maximum water level as configured by
/// the `TransformBuilder`.
///
/// # Output
/// The three methods of the `Watershed` trait each return a different set of
/// parameters based on the watershed transform of the input image:
/// - `transform` simply returns the watershed transform of the image.
/// - `transform_to_list` returns a list of the areas of all the lakes at each
/// water level. Its return type is`Vec<(u8, Vec<usize>)>`, where the `u8` equals
/// the water level and the `Vec<usize>` is the list of areas of all the lakes
/// at each water level. The `Vec<usize>` is the same length for each water level,
/// but may contain zero-sided lakes. The water levels are returned in order.
/// - `transform_history` returns a list of intermediate images of the watershed
/// transform at every water level. Its return type is `Vec<(u8, ndarray::Array2<usize>)`,
/// where the `u8` holds the water level that each `Array2` snapshot was taken at.
/// The water levels are returned in order.
///
/// # Artifacts and peculiarities
/// Due to some implementation details, the 1px-wide edges of the input array are
/// not accessible to the watershed transform. They will thus remain unfilled for
/// the entire duration of the transform.
///
/// A workaround can be enabled by calling `enable_edge_correction` on the
/// `TransformBuilder`. Enabling this setting copies the input image into a
/// new array, 1px wider on all sides. This padded array is then used as the
/// actual input to the watershed transform. The final output of the transform is
/// a copy of this intermediate array with the padding removed. The padding also
/// does not show up in the output of intermediate plots.
pub struct SegmentingWatershed<T = ()> {
  //Plot options
  #[cfg(feature = "plots")]
  plot_path: Option<std::path::PathBuf>,
  #[cfg(feature = "plots")]
  plot_colour_map:
    fn(count: usize, min: usize, max: usize) -> Result<RGBColor, Box<dyn std::error::Error>>,
  max_water_level: u8,
  edge_correction: bool,

  //Hooks
  wlvl_hook: Option<fn(HookCtx) -> T>,
}

impl<T> SegmentingWatershed<T> {
  fn clone_with_hook<U>(&self, hook: fn(HookCtx) -> U) -> SegmentingWatershed<U> {
    SegmentingWatershed {
      #[cfg(feature = "plots")]
      plot_path: self.plot_path.clone(),
      #[cfg(feature = "plots")]
      plot_colour_map: self.plot_colour_map,
      max_water_level: self.max_water_level,
      edge_correction: self.edge_correction,
      wlvl_hook: Some(hook),
    }
  }
}

impl<T> Watershed<T> for SegmentingWatershed<T> {
  fn transform_with_hook(&self, input: nd::ArrayView2<u8>, seeds: &[(usize, usize)]) -> Vec<T> {
    //(1a) make an image for holding the diffserent water colours
    let shape = if self.edge_correction {
      //If the edge correction is enabled, we have to pad the input with a 1px
      //wide border, which increases the size shape of the output image by two
      [input.shape()[0] + 2, input.shape()[1] + 2]
    } else {
      [input.shape()[0], input.shape()[1]]
    };
    let mut output = nd::Array2::<usize>::zeros(shape);

    //(1b) reshape the input image if necessary
    let mut padded_input =
      if self.edge_correction { Some(nd::Array2::<u8>::zeros(shape)) } else { None };
    let input = if self.edge_correction {
      //Copy the input pixel values into the new padded image
      nd::Zip::from(
        padded_input
          .as_mut()
          .expect("corrected_input was None, which should be impossible. Please report this bug.")
          .slice_mut(nd::s![1..(shape[0] - 1), 1..(shape[1] - 1)]),
      )
      .and(input)
      .into_par_iter()
      .for_each(|(a, &b)| *a = b);
      padded_input.as_ref().unwrap().view()
    } else {
      input.reborrow()
    };

    //(2) set "colours" for each of the starting points
    // The colours should range from 1 to seeds.len()
    let mut colours: Vec<usize> = (1..=seeds.len()).into_iter().collect();
    let seed_colours: Vec<_> =
      colours.iter().zip(seeds.iter()).map(|(col, (x, z))| (*col, (*x, *z))).collect();

    //Colour the starting pixels
    for (&idx, &col) in seeds.iter().zip(colours.iter()) {
      output[idx] = col;
    }
    //Set the zeroth colour to UNCOLOURED!
    colours.insert(UNCOLOURED, UNCOLOURED);

    #[cfg(feature = "debug")]
    println!("starting with {} lakes", colours.len());

    //(3) set-up progress bar
    #[cfg(feature = "progress")]
    let bar = set_up_bar(self.max_water_level);

    //(4) increase water level to specified maximum
    (0..=self.max_water_level)
      .into_iter()
      .map(|water_level| {
        //(logging) make a new perfreport
        #[cfg(feature = "debug")]
        let mut perf = crate::performance_monitoring::PerfReport::default();
        #[cfg(feature = "debug")]
        let loop_start = std::time::Instant::now();

        /*(i) Colour all flooded pixels connected to a source
          We have to loop multiple times because there may be plateau's. These
          require us to colour more than just one neighbouring pixel -> we need
          to loop until there are no more uncoloured, flooded pixels connected to
          a source left.
        */
        'colouring_loop: loop {
          #[cfg(feature = "progress")]
          {
            bar.tick(); //Tick the progressbar
          }
          #[cfg(feature = "debug")]
          {
            perf.loops += 1;
          }

          #[cfg(feature = "debug")]
          let iter_start = std::time::Instant::now();

          /*(A) Find pixels to colour this iteration
            We first look for all pixels that are uncoloured, flooded and directly
            attached to a coloured pixel. We do this in parallel. We cannot, however,
            change the pixel colours *and* look for pixels to colour at the same time.
            That is why we collect all pixels to colour in a vector, and later update
            the map.
          */
          let pix_to_colour = find_flooded_px(input.view(), output.view(), water_level);

          #[cfg(feature = "debug")]
          perf.big_iter_ms.push(iter_start.elapsed().as_millis() as usize);

          /*(B) Colour pixels that we found in step (A)
            If there are no pixels to be coloured anymore, we can break from this
            loop and raise the water level
          */
          if pix_to_colour.is_empty() {
            //No more connected, flooded pixels left -> raise water level
            break 'colouring_loop;
          } else {
            //We have pixels to colour
            #[cfg(feature = "debug")]
            let colour_start = std::time::Instant::now();

            pix_to_colour.into_iter().for_each(|(idx, col)| {
              output[idx] = col;
            });

            #[cfg(feature = "debug")]
            perf.colouring_mus.push(colour_start.elapsed().as_micros() as usize);
          }
        }

        /* (ii) Merge all touching regions
          We do not do this for the segmenting transform!
        */
        #[cfg(feature = "debug")]
        {
          perf.merge_ms = 0;
        }

        //(iii) Plot current state of the watershed transform
        #[cfg(feature = "plots")]
        if let Some(ref path) = self.plot_path {
          if let Err(err) = plotting::plot_slice(
            if self.edge_correction {
              //Do not plot the edge correction padding
              output.slice(nd::s![1..(shape[0] - 1), 1..(shape[1] - 1)])
            } else {
              output.view()
            },
            &path.join(&format!("ws_lvl{water_level}.png")),
            self.plot_colour_map,
          ) {
            println!("Could not make watershed plot. Error: {err}")
          }
        }

        //(iv) print performance report
        #[cfg(all(feature = "debug", feature = "progress"))]
        {
          //In this combination we have a progress bar, we should use it to print
          perf.total_ms = loop_start.elapsed().as_millis() as usize;
          bar.println(format!("{perf}"));
        }
        #[cfg(all(feature = "debug", not(feature = "progress")))]
        {
          //We do not have a progress bar, so a plain println! will have to do
          perf.total_ms = loop_start.elapsed().as_millis() as usize;
          println!("{perf}");
        }

        //(v) Update progressbar and plot stuff
        #[cfg(feature = "progress")]
        {
          bar.inc(1);
        }

        //(vi) Execute hook (if one is provided)
        self.wlvl_hook.and_then(|hook| {
          Some(hook(HookCtx::ctx(
            water_level,
            self.max_water_level,
            input.view(),
            output.view(),
            &seed_colours,
          )))
        })
      })
      .filter_map(|x| x)
      .collect()
  }

  fn transform(&self, input: nd::ArrayView2<u8>, seeds: &[(usize, usize)]) -> nd::Array2<usize> {
    //(1) Make a copy of self with the appropriate hook
    let proper_transform = self.clone_with_hook(|ctx| {
      if ctx.water_level == ctx.max_water_level {
        Some(ctx.colours.to_owned())
      } else {
        None
      }
    });

    //(2) Perform transform with new hook
    proper_transform.transform_with_hook(input, seeds)[0].as_ref().expect("no output?").clone()
  }

  fn transform_history(
    &self,
    input: nd::ArrayView2<u8>,
    seeds: &[(usize, usize)],
  ) -> Vec<(u8, nd::Array2<usize>)> {
    //(1) Make a copy of self with the appropriate hook
    let proper_transform =
      self.clone_with_hook(|ctx| (ctx.water_level, ctx.colours.to_owned()));

    //(2) Perform transform with new hook
    proper_transform.transform_with_hook(input, seeds)
  }

  fn transform_to_list(
    &self,
    input: nd::ArrayView2<u8>,
    seeds: &[(usize, usize)],
  ) -> Vec<(u8, Vec<usize>)> {
    //(1) Make a copy of self with the appropriate hook
    let proper_transform = self.clone_with_hook(find_lake_sizes);

    //(2) Perform transform with new hook
    proper_transform.transform_with_hook(input, seeds)
  }

}
