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
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rustronomy_watershed::prelude::*;

#[test]
fn core_bench() {
  //Create a random uniform distribution
  let rf = nd::Array2::<u8>::random((1024, 1024), Uniform::new(0, 254));

  //Set-up the watershed transform
  let watershed = TransformBuilder::new_merging().build().unwrap();

  //Find minima of the random field (to be used as seeds)
  let rf_mins = watershed.find_local_minima(rf.view());

  println!("Testing 1 to {} threads performance", rayon::current_num_threads());

  //Time with num cores
  let results: Vec<f64> = (1..=rayon::current_num_threads())
    .into_iter()
    .map(|num_threads| {
      //Set core count
      println!("Running algorithm with {num_threads} thread(s)");
      let pool = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
      //Time watershed
      let start = std::time::Instant::now();
      pool.install(|| watershed.transform_to_list(rf.view(), &rf_mins));
      start.elapsed().as_secs_f64()
    })
    .collect();

  //Print per run results
  for (threads, time) in results.iter().enumerate().map(|(i, t)| (i + 1, t)) {
    println!("{threads:02} threads = {time:000.02}s");
  }

  //Print total results
  let average = (1.0 / (results.len() as f64)) * results.iter().sum::<f64>();
  println!("Average time: {average:.02}");
}
