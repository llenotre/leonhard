#![allow(dead_code)]

mod complex;
mod linear_algebra;
mod polynomial;

pub trait Field<T>: Default + Copy
	+ std::ops::Neg<Output = T>
	+ std::ops::Add<Output = T>
	+ std::ops::AddAssign
	+ std::ops::Sub<Output = T>
	+ std::ops::SubAssign
	+ std::ops::Mul<Output = T>
	+ std::ops::MulAssign
	+ std::ops::Div<Output = T>
	+ std::ops::DivAssign {

	fn additive_identity() -> T;
	fn multiplicative_identity() -> T;

	fn sqrt(self) -> T;
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

			fn sqrt(self) -> $type {
				f64::sqrt(self as f64) as $type
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

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_mat_add() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);
		mat += 1.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq!(*mat.get(i, j), 1. as f64);
			}
		}
	}

	// TODO Subtract
	// TODO Multiply
	// TODO Matrix Multiply
	// TODO Vector transform
	// TODO Divide

	// TODO Transpose
	// TODO Row Echelon
	// TODO Determinant
	// TODO Inverse

	#[test]
	fn test_mat_trace() {
		let mut mat = linear_algebra::Matrix::<f64>::new(4, 3);
		mat += 1.;
		assert_eq!(mat.trace(), 3. as f64);
	}

	// TODO Vector length squared
	// TODO Vector length
	// TODO Vector normalize
	// TODO Vector dot
	// TODO Vector cross product

	// TODO Polynom

	// TODO Complex
}
