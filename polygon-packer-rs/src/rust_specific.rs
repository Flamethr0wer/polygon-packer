//! this module handles rust-specific things like
//! hot-swapping the float type to squeeze out more perf.

use std::f32;
use std::f64;

/// change this type to `f32` be less precise but get faster,
/// or `f64` to be more precise but get slower.
pub type FloatType = f32;

/// an interface that defines PI for both 32 bit and 64 bit floats.
pub trait AssocPI {
    const PI: Self;
}

// associated PI constant for f32
impl AssocPI for f32 {
    const PI: f32 = f32::consts::PI;
}

// associated PI constant for f64
impl AssocPI for f64 {
    const PI: f64 = f64::consts::PI;
}
