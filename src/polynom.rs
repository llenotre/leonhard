use crate::Field;

fn compute<T: Field<T>>(c: Vec<T>, x: T) -> T {
	if c.len() == 0 {
		return T::additive_identity();
	}

	let mut x_ = x;
	let mut n = c[0];
	for i in 1..c.len() {
		n += c[i] * x_;
		x_ *= x_;
	}
	n
}
