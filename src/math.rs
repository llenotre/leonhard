pub fn lerp<T>(v0: T, v1: T, t: f64) -> T
	where T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T> {
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
