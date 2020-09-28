use crate::Field;

pub fn compute<T: Field<T>>(c: Vec<T>, x: T) -> T {
	if c.len() == 0 {
		return T::additive_identity();
	}

	let mut n = c[0];
	let mut x_ = x;
	for i in 1..c.len() {
		n += c[i] * x_;
		x_ *= x;
	}
	n
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_poly_compute0() {
		assert_eq!(compute::<f64>(vec!{3., 2., 1.}, 1.), 6. as f64);
		assert_eq!(compute::<f64>(vec!{3., 2., 1.}, -1.), 2. as f64);
	}

	#[test]
	fn test_poly_compute1() {
		assert_eq!(compute::<f64>(vec!{4., 3., 2., 1.}, 1.), 10. as f64);
		assert_eq!(compute::<f64>(vec!{4., 3., 2., 1.}, -1.), 2. as f64);
	}

	#[test]
	fn test_poly_compute2() {
		assert_eq!(compute::<f64>(vec!{1., 2., 0., 0.8}, 1.), 3.8 as f64);
		assert_eq!(compute::<f64>(vec!{1., 2., 0., 0.8}, -1.), -1.8 as f64);
	}
}
