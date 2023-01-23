![rustronomy_dark_banner](https://github.com/smups/rustronomy/blob/main/logos/Rustronomy-watershed_github_banner_dark.png?raw=true#gh-light-mode-only)
![rustronomy_light_banner](https://github.com/smups/rustronomy/blob/main/logos/Rustronomy-watershed_github_banner_light.png#gh-dark-mode-only)
# rustronomy-watershed changelog

## v0.2.0
Updated `Watershed` trait (public API breaking change)
- `transform` method now returns actual watershed transform of the image
- the old `transform` method has been renamed to `transform_to_list`
- added new `transform_history` method that keeps track of the intermediate state
of the transform

## v0.1.0 Initial release