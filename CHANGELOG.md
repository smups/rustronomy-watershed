![](https://github.com/smups/rustronomy/blob/main/logos/Rustronomy-watershed_github_banner_dark.png?raw=true#gh-light-mode-only)
![](https://github.com/smups/rustronomy/blob/main/logos/Rustronomy-watershed_github_banner_light.png#gh-dark-mode-only)
# rustronomy-watershed changelog

## v0.4.0
*This version breaks the currently existing API*

This version adds the ability to run custom code each time the water level is
raised during the watershed transformation. This enables running customizable 
code for data analysis purposes.

Unfortunately, this feature addition required the following breaking API changes:
- The `TransformBuilder` struct now has a generic field
- The `TransformBuilder` struct no longer has `build`, `new_segmenting` and
`new_merging` methods. They have been replaced by `new`, `default`
`build_segmenting` and `build_merging`
- The `Watershed` trait is no longer dyn-safe and also has a generic field

The generic parameter in the new `Watershed` trait is the return type of the new
custom hook that is ran each time the water level is raised. It has the default
value `()`. Sadly rust is currently not able to infer the type of the generic
parameter unless it is explicitly specified by using such a hook. To get around
this limitation, you can use the `default()` method to start configuring the 
`TransformBuilder`, which explicitly specifies the default type.

## v0.3.2
_This version adds a new feature, but does not break the existing API and is
therefore marked as minor._

This version adds a new option to the `TransformBuilder`: `enable_edge_correction()`.
Calling this method enables a correction to the watershed algorithm which fixes
the edges of input images never being included in the watershed transform. This
incurs a performance and memory penalty, since the input image has to be embedded
in a larger canvas.

The shape of intermediate plots (size in pixels) and the output of the transform
are not effected by this change.

## v0.3.1
Removed a `println!` debug statement that I forgot about 

## v0.3.0
This version does not contain any breaking API changes, but it fixes a *major*
bug in the merging watershed transforms. Previously, there was some randomisation
of the "colours" of certain lakes which caused random lakes to merge, even if 
they were not touching. This has been fixed.

In addition, a new utility function has been added: `pre_processor_with_max`. This function can be used as a replacement for the pre-processor if you want to normalise the input to the water transform to a different range than `1..u8::MAX-1`, like so:
```rust
use rustronomy_watershed::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
///
//Set custom maximum waterlevel
const MY_MAX: u8 = 127;

//Create a random uniform distribution
let rf = nd::Array2::<f64>::random((512, 512), Uniform::new(0.0, 1.0));

//Set-up the watershed transform
let watershed = TransformBuilder::new_segmenting()
    .set_max_water_lvl(MY_MAX)
    .build()
    .unwrap();

//Run pre-processor (using turbofish syntax)
let rf = watershed.pre_processor_with_max::<{MYMAX}, _, _>(rf.view());

//Find minima of the random field (to be used as seeds)
let rf_mins = watershed.find_local_minima(rf.view());
//Execute the watershed transform
let output = watershed.transform(rf.view(), &rf_mins)
```
The braces in `{MYMAX}` are necessary for the code to compile. This is an unfortunate limitation of rustc, which may get fixed in the future. For now, just use curly braces!

## v0.2.0
Updated `Watershed` trait (public API breaking change)
- `transform` method now returns actual watershed transform of the image
- the old `transform` method has been renamed to `transform_to_list`
- added new `transform_history` method that keeps track of the intermediate state
of the transform

## v0.1.0 Initial release