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
