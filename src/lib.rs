#![allow(dead_code)]

mod complex;
mod linear_algebra;
mod polynom;

pub trait Field<T>: Copy
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

	fn sqrt(&self) -> T;
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

			fn sqrt(&self) -> $type {
				f64::sqrt(*self as f64) as $type
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

pub fn lerp<T>(v0: &T, v1: &T, t: f64) -> T
	where T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> {
	*v0 * (1. - t) + *v1 * t
}

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

	#[test]
	fn test_mat_sub() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);
		mat -= 1.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq!(*mat.get(i, j), -1. as f64);
			}
		}
	}

	#[test]
	fn test_mat_mul() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);
		for i in 0..3 {
			for j in 0..3 {
				*mat.get_mut(i, j) = (i * 3 + j) as f64;
			}
		}
		mat *= 2.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq!(*mat.get(i, j), (i * 3 + j) as f64 * 2.);
			}
		}
	}

	// TODO Matrix Multiply
	// TODO Vector transform

	#[test]
	fn test_mat_div() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);
		for i in 0..3 {
			for j in 0..3 {
				*mat.get_mut(i, j) = (i * 3 + j) as f64;
			}
		}
		mat /= 2.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq!(*mat.get(i, j), (i * 3 + j) as f64 / 2.);
			}
		}
	}

	#[test]
	fn test_mat_transpose() {
		let mut mat = linear_algebra::Matrix::<f64>::new(4, 3);

		assert!(!mat.is_transposed());
		assert!(mat.get_height() == 4);
		assert!(mat.get_width() == 3);
		*mat.get_mut(0, 0) = 1.;
		*mat.get_mut(1, 0) = 2.;
		*mat.get_mut(1, 2) = 3.;

		mat.transpose();

		assert!(mat.is_transposed());
		assert!(mat.get_height() == 3);
		assert!(mat.get_width() == 4);
		assert_eq!(*mat.get(0, 0), 1.);
		assert_eq!(*mat.get(0, 1), 2.);
		assert_eq!(*mat.get(2, 1), 3.);
		assert_eq!(*mat.get(1, 0), 0.);
		assert_eq!(*mat.get(1, 2), 0.);
	}

	#[test]
	fn test_mat_row_echelon0() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);
		mat.to_row_echelon();

		for i in 0..mat.get_height() {
			for j in 0..mat.get_width() {
				assert_eq!(*mat.get(i, j), 0.);
			}
		}
	}

	#[test]
	fn test_mat_row_echelon1() {
		let mut mat = linear_algebra::Matrix::<f64>::identity(3);
		mat.to_row_echelon();

		for i in 0..mat.get_height() {
			for j in 0..mat.get_width() {
				assert_eq!(*mat.get(i, j), if i == j { 1. } else { 0. });
			}
		}
	}

	#[test]
	fn test_mat_row_echelon2() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);
		*mat.get_mut(0, 0) = 0.;
		*mat.get_mut(0, 1) = 1.;
		*mat.get_mut(0, 2) = 2.;
		*mat.get_mut(1, 0) = 3.;
		*mat.get_mut(1, 1) = 4.;
		*mat.get_mut(1, 2) = 5.;
		*mat.get_mut(2, 0) = 6.;
		*mat.get_mut(2, 1) = 7.;
		*mat.get_mut(2, 2) = 8.;

		mat.to_row_echelon();

		assert_eq!(*mat.get(0, 0), 1.);
		assert_eq!(*mat.get(0, 1), 0.);
		assert_eq!(*mat.get(0, 2), -1.);
		assert_eq!(*mat.get(1, 0), 0.);
		assert_eq!(*mat.get(1, 1), 1.);
		assert_eq!(*mat.get(1, 2), 2.);
		assert_eq!(*mat.get(2, 0), 0.);
		assert_eq!(*mat.get(2, 1), 0.);
		assert_eq!(*mat.get(2, 2), 0.);
	}

	#[test]
	fn test_mat_determinant0() {
		let mat = linear_algebra::Matrix::<f64>::new(3, 3);
		assert_eq!(mat.determinant(), 0 as f64);
	}

	#[test]
	fn test_mat_determinant1() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);

		*mat.get_mut(0, 0) = -2.;
		*mat.get_mut(0, 1) = 2.;
		*mat.get_mut(0, 2) = -3.;
		*mat.get_mut(1, 0) = -1.;
		*mat.get_mut(1, 1) = 1.;
		*mat.get_mut(1, 2) = 3.;
		*mat.get_mut(2, 0) = 2.;
		*mat.get_mut(2, 1) = 0.;
		*mat.get_mut(2, 2) = -1.;

		assert_eq!(mat.determinant(), 18 as f64);
	}

	#[test]
	fn test_mat_determinant2() {
		let mut mat = linear_algebra::Matrix::<f64>::new(3, 3);

		*mat.get_mut(0, 0) = -2.;
		*mat.get_mut(0, 1) = 2.;
		*mat.get_mut(0, 2) = -3.;
		*mat.get_mut(1, 0) = 0.;
		*mat.get_mut(1, 1) = 2.;
		*mat.get_mut(1, 2) = -4.;
		*mat.get_mut(2, 0) = 0.;
		*mat.get_mut(2, 1) = 0.;
		*mat.get_mut(2, 2) = 4.5;

		assert_eq!(mat.determinant(), -18 as f64);
	}

	// TODO Inverse
	// TODO Rank

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

	#[test]
	fn test_poly_compute0() {
		assert_eq!(polynom::compute::<f64>(vec!{3., 2., 1.}, 1.), 6. as f64);
		assert_eq!(polynom::compute::<f64>(vec!{3., 2., 1.}, -1.), 2. as f64);
	}

	#[test]
	fn test_poly_compute1() {
		assert_eq!(polynom::compute::<f64>(vec!{4., 3., 2., 1.}, 1.), 10. as f64);
		assert_eq!(polynom::compute::<f64>(vec!{4., 3., 2., 1.}, -1.), 2. as f64);
	}

	#[test]
	fn test_poly_compute2() {
		assert_eq!(polynom::compute::<f64>(vec!{1., 2., 0., 0.8}, 1.), 3.8 as f64);
		assert_eq!(polynom::compute::<f64>(vec!{1., 2., 0., 0.8}, -1.), -1.8 as f64);
	}

	// TODO Complex
}
