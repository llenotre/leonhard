#![allow(dead_code)]

mod complex;
mod linear_algebra;
mod math;
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

	fn mul_add(&self, a: &T, b: &T) -> T;

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

			fn mul_add(&self, a: &$type, b: &$type) -> $type {
				f64::mul_add(*self as f64, *a as f64, *b as f64) as $type
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

	#[test]
	fn test_mat_mul_mat0() {
		let mat0 = linear_algebra::Matrix::<f64>::identity(3);
		let mat1 = linear_algebra::Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,
		});
		let mat2 = mat0 * mat1;

		assert_eq!(*mat2.get(0, 0), 0. as f64);
		assert_eq!(*mat2.get(0, 1), 1. as f64);
		assert_eq!(*mat2.get(0, 2), 2. as f64);
		assert_eq!(*mat2.get(1, 0), 3. as f64);
		assert_eq!(*mat2.get(1, 1), 4. as f64);
		assert_eq!(*mat2.get(1, 2), 5. as f64);
		assert_eq!(*mat2.get(2, 0), 6. as f64);
		assert_eq!(*mat2.get(2, 1), 7. as f64);
		assert_eq!(*mat2.get(2, 2), 8. as f64);
	}

	#[test]
	fn test_mat_mul_mat1() {
		let mat0 = linear_algebra::Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,
		});
		let mat2 = mat0.clone() * mat0;

		assert_eq!(*mat2.get(0, 0), 15. as f64);
		assert_eq!(*mat2.get(0, 1), 18. as f64);
		assert_eq!(*mat2.get(0, 2), 21. as f64);
		assert_eq!(*mat2.get(1, 0), 42. as f64);
		assert_eq!(*mat2.get(1, 1), 54. as f64);
		assert_eq!(*mat2.get(1, 2), 66. as f64);
		assert_eq!(*mat2.get(2, 0), 69. as f64);
		assert_eq!(*mat2.get(2, 1), 90. as f64);
		assert_eq!(*mat2.get(2, 2), 111. as f64);
	}

	#[test]
	fn test_mat_mul_vec0() {
		let mat = linear_algebra::Matrix::<f64>::identity(3);
		let vec0 = linear_algebra::Vector::<f64>::from_vec(vec!{
			0., 1., 2.,
		});
		let vec1 = mat * vec0;

		assert_eq!(*vec1.get(0), 0. as f64);
		assert_eq!(*vec1.get(1), 1. as f64);
		assert_eq!(*vec1.get(2), 2. as f64);
	}

	#[test]
	fn test_mat_mul_vec1() {
		let mat = linear_algebra::Matrix::<f64>::identity(3) * 2.;
		let vec0 = linear_algebra::Vector::<f64>::from_vec(vec!{
			0., 1., 2.,
		});
		let vec1 = mat * vec0;

		assert_eq!(*vec1.get(0), 0. as f64);
		assert_eq!(*vec1.get(1), 2. as f64);
		assert_eq!(*vec1.get(2), 4. as f64);
	}

	#[test]
	fn test_mat_mul_vec2() {
		let mat = linear_algebra::Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 0.,
			1., 0., 1.,
			0., 1., 0.,
		});
		let vec0 = linear_algebra::Vector::<f64>::from_vec(vec!{
			0., 1., 2.,
		});
		let vec1 = mat * vec0;

		assert_eq!(*vec1.get(0), 1. as f64);
		assert_eq!(*vec1.get(1), 2. as f64);
		assert_eq!(*vec1.get(2), 1. as f64);
	}

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
		let mut mat = linear_algebra::Matrix::<f64>::from_vec(4, 3, vec!{
			1., 0., 0.,
			2., 0., 3.,
			0., 0., 0.,
			0., 0., 0.,
		});

		assert!(!mat.is_transposed());
		assert!(mat.get_height() == 4);
		assert!(mat.get_width() == 3);
		assert_eq!(*mat.get(0, 0), 1.);
		assert_eq!(*mat.get(1, 0), 2.);
		assert_eq!(*mat.get(1, 2), 3.);
		assert_eq!(*mat.get(0, 1), 0.);
		assert_eq!(*mat.get(2, 1), 0.);

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
		let mut mat = linear_algebra::Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,
		});

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
		let mat = linear_algebra::Matrix::<f64>::identity(3);
		assert_eq!(mat.determinant(), 1 as f64);
	}

	#[test]
	fn test_mat_determinant2() {
		let mat = linear_algebra::Matrix::<f64>::from_vec(3, 3, vec!{
			-2., 2., -3.,
			-1., 1., 3.,
			2., 0., -1.,
		});

		assert_eq!(mat.determinant(), 18 as f64);
	}

	#[test]
	fn test_mat_determinant3() {
		let mat = linear_algebra::Matrix::<f64>::from_vec(3, 3, vec!{
			-2., 2., -3.,
			0., 2., -4.,
			0., 0., 4.5,
		});

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

	#[test]
	fn test_vec_length0() {
		let vec = linear_algebra::Vector::<f64>::from_vec(vec!{1., 0., 0.});
		assert_eq!(vec.length(), 1. as f64);
	}

	#[test]
	fn test_vec_length1() {
		let vec = linear_algebra::Vector::<f64>::from_vec(vec!{1., 1., 1.});
		assert_eq!(vec.length(), (3. as f64).sqrt());
	}

	#[test]
	fn test_vec_normalize0() {
		let mut vec = linear_algebra::Vector::<f64>::from_vec(vec!{1., 1., 1.});
		vec.normalize();
		assert_eq!(vec.length(), 1. as f64);
	}

	#[test]
	fn test_vec_dot0() {
		let vec0 = linear_algebra::Vector::<f64>::from_vec(vec!{1., 0., 0.});
		let vec1 = linear_algebra::Vector::<f64>::from_vec(vec!{0., 1., 0.});
		assert_eq!(vec0.dot(&vec1), 0. as f64);
	}

	#[test]
	fn test_vec_cross_product0() {
		let vec0 = linear_algebra::Vector::<f64>::from_vec(vec!{1., 0., 0.});
		let vec1 = linear_algebra::Vector::<f64>::from_vec(vec!{0., 1., 0.});
		let vec2 = vec0.cross_product(&vec1);
		assert_eq!(*vec2.get(0), 0. as f64);
		assert_eq!(*vec2.get(1), 0. as f64);
		assert_eq!(*vec2.get(2), 1. as f64);
	}

	#[test]
	fn test_vec_cross_product1() {
		let vec0 = linear_algebra::Vector::<f64>::from_vec(vec!{1., 0.5, 2.});
		let vec1 = linear_algebra::Vector::<f64>::from_vec(vec!{0.8, 1., 0.});
		let vec2 = vec0.cross_product(&vec1);
		assert_eq!(*vec2.get(0), -2. as f64);
		assert_eq!(*vec2.get(1), 1.6 as f64);
		assert_eq!(*vec2.get(2), 0.6 as f64);
	}

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

	#[test]
	fn test_complex_mul0() {
		let c = complex::Complex::<f64>::new(&1., &0.) * 2.;
		assert_eq!(c.x, 2 as f64);
		assert_eq!(c.y, 0 as f64);
	}

	#[test]
	fn test_complex_mul1() {
		let c = complex::Complex::<f64>::new(&1., &1.) * 2.;
		assert_eq!(c.x, 2 as f64);
		assert_eq!(c.y, 2 as f64);
	}

	#[test]
	fn test_complex_mul2() {
		let c0 = complex::Complex::<f64>::new(&1., &1.);
		let c1 = complex::Complex::<f64>::new(&2., &2.);
		let c2 = c0 * c1;
		assert_eq!(c2.x, 0 as f64);
		assert_eq!(c2.y, 4 as f64);
	}

	#[test]
	fn test_complex_div0() {
		let c = complex::Complex::<f64>::new(&1., &0.) / 2.;
		assert_eq!(c.x, 0.5 as f64);
		assert_eq!(c.y, 0 as f64);
	}

	#[test]
	fn test_complex_div1() {
		let c = complex::Complex::<f64>::new(&1., &1.) / 2.;
		assert_eq!(c.x, 0.5 as f64);
		assert_eq!(c.y, 0.5 as f64);
	}

	#[test]
	fn test_complex_div2() {
		let c0 = complex::Complex::<f64>::new(&1., &1.);
		let c1 = complex::Complex::<f64>::new(&2., &2.);
		let c2 = c0 / c1;
		assert_eq!(c2.x, 0.5 as f64);
		assert_eq!(c2.y, 0. as f64);
	}

	#[test]
	fn test_lerp0() {
		assert_eq!(math::lerp::<f64>(10., 15., 0.), 10 as f64);
		assert_eq!(math::lerp::<f64>(10., 15., 1.), 15 as f64);
		assert_eq!(math::lerp::<f64>(10., 15., 0.5), 12.5 as f64);
	}

	#[test]
	fn test_lerp1() {
		assert_eq!(math::lerp::<f64>(10., -10., 0.), 10 as f64);
		assert_eq!(math::lerp::<f64>(10., -10., 1.), -10 as f64);
		assert_eq!(math::lerp::<f64>(10., -10., 0.5), 0 as f64);
	}

	#[test]
	fn test_binomial_coefficient0() {
		assert_eq!(math::binomial_coefficient(5, 2), 10);
		assert_eq!(math::binomial_coefficient(50, 2), 1225);
		assert_eq!(math::binomial_coefficient(8, 4), 70);
		assert_eq!(math::binomial_coefficient(80, 4), 1581580);
	}

	#[test]
	fn test_binomial_coefficient1() {
		for i in 0..100 {
			assert_eq!(math::binomial_coefficient(i, i), 1);
		}
	}

	#[test]
	fn test_binomial_coefficient2() {
		for i in 6..100 {
			assert_eq!(math::binomial_coefficient(5, i), 0);
		}
	}
}
