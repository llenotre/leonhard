pub fn lerp<T>(v0: T, v1: T, t: f64) -> T
	where T: std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> {
	v0 * (1. - t) + v1 * t
}

// TODO Generic
pub fn binomial_coefficient(n: usize, k: usize) -> usize {
	let mut k_ = k;

	if k > n {
		return 0;
	}
	if k * 2 > n {
		k_ = n - k;
	}

	let mut n_ = n;
	let mut c: usize = 1;

	for i in 1..=k_ {
		c = (c / i) * n_ + (c % i) * n_ / i;
		n_ -= 1;
	}
	c
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_lerp0() {
		assert_eq!(lerp::<f64>(10., 15., 0.), 10 as f64);
		assert_eq!(lerp::<f64>(10., 15., 1.), 15 as f64);
		assert_eq!(lerp::<f64>(10., 15., 0.5), 12.5 as f64);
	}

	#[test]
	fn test_lerp1() {
		assert_eq!(lerp::<f64>(10., -10., 0.), 10 as f64);
		assert_eq!(lerp::<f64>(10., -10., 1.), -10 as f64);
		assert_eq!(lerp::<f64>(10., -10., 0.5), 0 as f64);
	}

	#[test]
	fn test_binomial_coefficient0() {
		assert_eq!(binomial_coefficient(0, 1), 0);
		assert_eq!(binomial_coefficient(1, 0), 1);

		assert_eq!(binomial_coefficient(5, 2), 10);
		assert_eq!(binomial_coefficient(50, 2), 1225);
		assert_eq!(binomial_coefficient(8, 4), 70);
		assert_eq!(binomial_coefficient(80, 4), 1581580);
	}

	#[test]
	fn test_binomial_coefficient1() {
		for i in 0..100 {
			assert_eq!(binomial_coefficient(i, i), 1);
		}
	}

	#[test]
	fn test_binomial_coefficient2() {
		for i in 1..100 {
			assert_eq!(binomial_coefficient(0, i), 0);
		}
	}

	#[test]
	fn test_binomial_coefficient3() {
		for i in 0..100 {
			for j in (i + 1)..100 {
				assert_eq!(binomial_coefficient(i, j), 0);
			}
		}
	}
}
