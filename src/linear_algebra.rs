use std::cmp::min;
use crate::Value;

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
	transposed: bool,
}

impl<T: Value<T>> Matrix::<T> {
	pub fn new(height: usize, width: usize) -> Self {
		// TODO Check that `width` and `height` are not `0`
		let mut mat = Self {
			height: height,
			width: width,
			data: Vec::with_capacity(height * width),
			transposed: false,
		};
		mat.data.resize(height * width, T::default());
		mat
	}

	pub fn get_height(&self) -> usize {
		if self.transposed {
			self.width
		} else {
			self.height
		}
	}

	pub fn get_width(&self) -> usize {
		if self.transposed {
			self.height
		} else {
			self.width
		}
	}

	pub fn is_square(&self) -> bool {
		self.height == self.width
	}

	pub fn is_transposed(&self) -> bool {
		self.transposed
	}

	pub fn get(&self, y: usize, x: usize) -> &T {
		if self.transposed {
			&self.data[x * self.height + y]
		} else {
			&self.data[y * self.width + x]
		}
	}

	pub fn get_mut(&mut self, y: usize, x: usize) -> &mut T {
		if self.transposed {
			&mut self.data[x * self.height + y]
		} else {
			&mut self.data[y * self.width + x]
		}
	}

	/*pub fn submatrix(&self, y: usize, x: usize, height: usize, width: usize) -> Self {
		// TODO
	}*/

	pub fn transpose(&mut self) -> &mut Self {
		self.transposed = !self.transposed;
		self
	}

	pub fn to_row_echelon(&mut self) -> &mut Self {
		// TODO
		self
	}

	pub fn determinant(&self) -> T {
		// TODO
		T::default()
	}

	pub fn is_invertible() -> bool {
		// TODO
		true
	}

	pub fn inverse(&mut self) -> &mut Self {
		// TODO
		self
	}

	pub fn trace(&self) -> T {
		let max = min(self.get_height(), self.get_width());
		let mut n = T::default();

		for i in 0..max {
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
