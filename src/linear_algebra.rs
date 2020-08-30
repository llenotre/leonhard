use std::cmp::min;

trait Value: Default + Copy
	+ std::ops::AddAssign
	+ std::ops::SubAssign
	+ std::ops::MulAssign
	+ std::ops::DivAssign {}

trait Tensor<T: Value> {
	fn add_val(&mut self, n: &T);
	fn add_self(&mut self, n: &Self);

	fn subtract_val(&mut self, n: &T);
	fn subtract_self(&mut self, n: &Self);

	fn multiply_val(&mut self, n: &T);
	fn multiply_self(&mut self, n: &Self);

	fn divide_val(&mut self, n: &T);
	fn divide_self(&mut self, n: &Self);
}

#[derive(Clone)]
struct Matrix<T: Value> {
	pub height: usize,
	pub width: usize,
	data: Vec<T>,
}

impl<T: Value> Matrix::<T> {
	pub fn new(height: usize, width: usize) -> Self {
		// TODO Check that `width` and `height` are not `0`
		let mut mat = Self {
			height: height,
			width: width,
			data: Vec::with_capacity(height * width),
		};
		mat.data.resize(height * width, T::default());
		mat
	}

	pub fn is_square(&self) -> bool {
		self.height == self.width
	}

	pub fn get(&self, y: usize, x: usize) -> &T {
		&self.data[y * self.width + x]
	}

	pub fn get_mut(&mut self, y: usize, x: usize) -> &mut T {
		&mut self.data[y * self.width + x]
	}

	/*pub fn submatrix(&self, y: usize, x: usize, height: usize, width: usize) -> Self {
		// TODO
	}

	pub fn transpose(&self) -> Self {
		// TODO
	}

	pub fn inverse(&self) -> Self {
		// TODO
	}

	pub fn determinant(&self) -> T {
		// TODO
	}*/

	pub fn trace(&self) -> T {
		let mut n = T::default();

		for i in 0..min(self.height, self.width) {
			n += *self.get(i, i);
		}
		n
	}
}

impl<T: Value> Tensor::<T> for Matrix::<T> {
	fn add_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i += *n;
		}
	}

	fn add_self(&mut self, n: &Self) {
		for i in 0..min(self.data.len(), n.data.len()) {
			self.data[i] += n.data[i];
		}
	}

	fn subtract_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i -= *n;
		}
	}

	fn subtract_self(&mut self, n: &Self) {
		for i in 0..min(self.data.len(), n.data.len()) {
			self.data[i] -= n.data[i];
		}
	}

	fn multiply_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i *= *n;
		}
	}

	fn multiply_self(&mut self, n: &Self) {
		for i in 0..min(self.data.len(), n.data.len()) {
			self.data[i] *= n.data[i];
		}
	}

	fn divide_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i /= *n;
		}
	}

	fn divide_self(&mut self, n: &Self) {
		for i in 0..min(self.data.len(), n.data.len()) {
			self.data[i] /= n.data[i];
		}
	}
}

impl<T: Value> std::ops::Add<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn add(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.add_val(&n);
		m
	}
}

impl<T: Value> std::ops::AddAssign<T> for Matrix::<T> {
	fn add_assign(&mut self, n: T) {
		self.add_val(&n);
	}
}

impl<T: Value> std::ops::Sub<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn sub(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.subtract_val(&n);
		m
	}
}

impl<T: Value> std::ops::SubAssign<T> for Matrix::<T> {
	fn sub_assign(&mut self, n: T) {
		self.subtract_val(&n);
	}
}

impl<T: Value> std::ops::Mul<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn mul(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.multiply_val(&n);
		m
	}
}

impl<T: Value> std::ops::MulAssign<T> for Matrix::<T> {
	fn mul_assign(&mut self, n: T) {
		self.multiply_val(&n);
	}
}

impl<T: Value> std::ops::Div<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn div(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.divide_val(&n);
		m
	}
}

impl<T: Value> std::ops::DivAssign<T> for Matrix::<T> {
	fn div_assign(&mut self, n: T) {
		self.divide_val(&n);
	}
}

struct Vector<T: Value> {
	size: usize,
	data: Vec<T>,
}
