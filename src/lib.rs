#![allow(dead_code)]

pub mod complex;
pub mod linear_algebra;
pub mod math;
pub mod polynom;
pub mod statistics;

pub trait Field<T>: Copy + std::fmt::Display
	+ std::ops::Neg<Output = T>
	+ std::ops::Add<Output = T>
	+ std::ops::AddAssign
	+ std::ops::Sub<Output = T>
	+ std::ops::SubAssign
	+ std::ops::Mul<Output = T>
	+ std::ops::MulAssign
	+ std::ops::Div<Output = T>
	+ std::ops::DivAssign
	+ std::cmp::PartialOrd {

	fn additive_identity() -> T;
	fn multiplicative_identity() -> T;

	fn mul_add(&self, a: &T, b: &T) -> T;

	fn abs(&self) -> T;
	fn sqrt(&self) -> T;
	fn atan2(&self, n: &T) -> T;

	fn epsilon_equal(&self, n: &T) -> bool;
}

macro_rules! primitive_field {
	($type:ident) => {
		impl Field::<$type> for $type {
			fn additive_identity() -> $type {
				0 as $type
			}

			fn multiplicative_identity() -> $type {
				1 as $type
			}

			fn mul_add(&self, a: &$type, b: &$type) -> $type {
				f64::mul_add(*self as f64, *a as f64, *b as f64) as $type
			}

			fn abs(&self) -> $type {
				(*self as $type).abs()
			}

			fn sqrt(&self) -> $type {
				f64::sqrt(*self as f64) as $type
			}

			fn atan2(&self, n: &$type) -> $type {
				f64::atan2(*self as f64, *n as f64) as $type
			}

			fn epsilon_equal(&self, n: &$type) -> bool {
				((*n as f64) - (*self as f64)).abs() < 0.000001
			}
		}
	}
}

primitive_field!(i8);
primitive_field!(i16);
primitive_field!(i32);
primitive_field!(i64);
primitive_field!(f32);
primitive_field!(f64);
