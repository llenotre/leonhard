use crate::Value;

fn compute<T: Value<T>>(c: Vec<T>, x: T) -> T {
	if c.len() == 0 {
		return T::default();
	}

	let mut x_ = x;
	let mut n = c[0];
	for i in 1..c.len() {
		n += c[i] * x_;
		x_ *= x_;
	}
	n
}
