use std::cmp::min;

trait Value<T>: Default + Copy
	+ std::ops::Add<Output = T>
	+ std::ops::AddAssign
	+ std::ops::Sub<Output = T>
	+ std::ops::SubAssign
	+ std::ops::Mul<Output = T>
	+ std::ops::MulAssign
	+ std::ops::Div<Output = T>
	+ std::ops::DivAssign {}

trait Tensor<T: Value<T>> {
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
struct Matrix<T: Value<T>> {
	height: usize,
	width: usize,
	data: Vec<T>,
}

impl<T: Value<T>> Matrix::<T> {
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

	pub fn get_height(&self) -> usize {
		self.height
	}

	pub fn get_width(&self) -> usize {
		self.width
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

impl<T: Value<T>> Tensor::<T> for Matrix::<T> {
	fn add_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i += *n;
		}
	}

	fn add_self(&mut self, n: &Self) {
		// TODO Check other's size
		for i in 0..self.data.len() {
			self.data[i] += n.data[i];
		}
	}

	fn subtract_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i -= *n;
		}
	}

	fn subtract_self(&mut self, n: &Self) {
		// TODO Check other's size
		for i in 0..self.data.len() {
			self.data[i] -= n.data[i];
		}
	}

	fn multiply_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i *= *n;
		}
	}

	fn multiply_self(&mut self, n: &Self) {
		// TODO Check other's size
		for i in 0..self.data.len() {
			self.data[i] *= n.data[i];
		}
	}

	fn divide_val(&mut self, n: &T) {
		for i in &mut self.data {
			*i /= *n;
		}
	}

	fn divide_self(&mut self, n: &Self) {
		// TODO Check other's size
		for i in 0..self.data.len() {
			self.data[i] /= n.data[i];
		}
	}
}

impl<T: Value<T>> std::ops::Add<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn add(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.add_val(&n);
		m
	}
}

impl<T: Value<T>> std::ops::AddAssign<T> for Matrix::<T> {
	fn add_assign(&mut self, n: T) {
		self.add_val(&n);
	}
}

impl<T: Value<T>> std::ops::Sub<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn sub(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.subtract_val(&n);
		m
	}
}

impl<T: Value<T>> std::ops::SubAssign<T> for Matrix::<T> {
	fn sub_assign(&mut self, n: T) {
		self.subtract_val(&n);
	}
}

impl<T: Value<T>> std::ops::Mul<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn mul(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.multiply_val(&n);
		m
	}
}

impl<T: Value<T>> std::ops::MulAssign<T> for Matrix::<T> {
	fn mul_assign(&mut self, n: T) {
		self.multiply_val(&n);
	}
}

impl<T: Value<T>> std::ops::Div<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn div(self, n: T) -> Matrix::<T> {
		let mut m = self.clone();
		m.divide_val(&n);
		m
	}
}

impl<T: Value<T>> std::ops::DivAssign<T> for Matrix::<T> {
	fn div_assign(&mut self, n: T) {
		self.divide_val(&n);
	}
}

struct Vector<T: Value<T>> {
	size: usize,
	data: Vec<T>,
}

impl<T: Value<T>> Vector::<T> {
	pub fn get_size(&self) -> usize {
		self.size
	}

	pub fn length_squared(&self) -> T {
		let mut n = T::default();

		for i in 0..self.size {
			let v = self.data[i];
			n += v * v;
		}
		n
	}

	// TODO
	/*pub fn length(&self) -> T {
		self.length_squared().sqrt()
	}*/

	pub fn dot(&self, other: &Vector<T>) -> T {
		let mut n = T::default();

		// TODO Check other's size
		for i in 0..self.size {
			n += self.data[i] * other.data[i];
		}
		n
	}

	// TODO Cross product
}
