use std::cmp::min;

trait Tensor<T> {
	fn add_val(&mut self, n: &T);
	fn add(&mut self, n: &Self);
	fn subtract_val(&mut self, n: &T);
	fn subtract(&mut self, n: &Self);
	fn multiply_val(&mut self, n: &T);
	fn multiply(&mut self, n: &Self);
	fn divide_val(&mut self, n: &T);
	fn divide(&mut self, n: &Self);
}

struct Matrix<T> {
	height: usize,
	width: usize,
	data: Vec<T>,
}

struct Vector<T> {
	size: usize,
	data: Vec<T>,
}

impl<T> Matrix::<T> where T: Default + Copy + std::ops::AddAssign {
	pub fn new(height: usize, width: usize) -> Self {
		// TODO Check that `width` and `height` are not `0`
		Self {
			height: height,
			width: width,
			data: Vec::with_capacity(height * width),
		}
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

impl<T> Tensor::<T> for Matrix::<T> where T: Default + Copy + std::ops::AddAssign {
	fn add_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i += *n;
		}
	}

	fn add(&mut self, n: &Self) {
		// TODO
	}

	fn subtract_val(&mut self, n: &T) {
		// TODO
	}

	fn subtract(&mut self, n: &Self) {
		// TODO
	}

	fn multiply_val(&mut self, n: &T) {
		// TODO
	}

	fn multiply(&mut self, n: &Self) {
		// TODO
	}

	fn divide_val(&mut self, n: &T) {
		// TODO
	}

	fn divide(&mut self, n: &Self) {
		// TODO
	}
}
