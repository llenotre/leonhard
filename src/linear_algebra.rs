use std::cmp::min;
use crate::Field;

pub trait Tensor<T: Field<T>> {
	fn negate(&mut self);

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
pub struct Matrix<T: Field<T>> {
	height: usize,
	width: usize,
	data: Vec<T>,
	transposed: bool,
}

#[derive(Clone)]
pub struct Vector<T: Field<T>> {
	size: usize,
	data: Vec<T>,
}

impl<T: Field<T>> Matrix::<T> {
	pub fn new(height: usize, width: usize) -> Self {
		// TODO Check that `width` and `height` are not `0`
		let mut mat = Self {
			height: height,
			width: width,
			data: Vec::with_capacity(height * width),
			transposed: false,
		};
		mat.data.resize(height * width, T::additive_identity());
		mat
	}

	pub fn from_vec(height: usize, width: usize, values: Vec::<T>) -> Self {
		// TODO Check that `width` and `height` are not `0` and that `values`'s length corresponds
		let mat = Self {
			height: height,
			width: width,
			data: values,
			transposed: false,
		};
		mat
	}

	pub fn identity(size: usize) -> Matrix::<T> {
		let mut mat = Self::new(size, size);

		for i in 0..size {
			*mat.get_mut(i, i) = T::multiplicative_identity();
		}
		mat
	}

	pub fn get_height(&self) -> usize {
		if self.is_transposed() {
			self.width
		} else {
			self.height
		}
	}

	pub fn get_width(&self) -> usize {
		if self.is_transposed() {
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
			&self.data[x * self.width + y]
		} else {
			&self.data[y * self.width + x]
		}
	}

	pub fn get_mut(&mut self, y: usize, x: usize) -> &mut T {
		if self.transposed {
			&mut self.data[x * self.width + y]
		} else {
			&mut self.data[y * self.width + x]
		}
	}

	pub fn submatrix(&self, y: usize, x: usize, height: usize, width: usize) -> Self {
		// TODO Check that arguments are in range

		let mut m = Self::new(height, width);
		for i in 0..height {
			for j in 0..width {
				*m.get_mut(i, j) = *m.get(y + i, x + j);
			}
		}
		m
	}

	pub fn transpose(&mut self) -> &mut Self {
		self.transposed = !self.transposed;
		self
	}

	fn rows_swap(&mut self, i: usize, j: usize) {
		// TODO Check arguments
		for k in 0..self.get_width() {
			let tmp = *self.get(i, k);
			*self.get_mut(i, k) = *self.get(j, k);
			*self.get_mut(j, k) = tmp;
		}
	}

	fn to_row_echelon_(&mut self, d: &mut T) {
		let mut i = 0;
		let mut j = 0;

		while i < self.get_height() && j < self.get_width() {
			let i_max = {
				let mut max = i;

				for k in i..self.get_height() {
					if *self.get(k, j) > *self.get(max, j) {
						max = k;
					}
				}
				max
			};

			if *self.get(i_max, j) == T::additive_identity() {
				j += 1;
				continue;
			}

			self.rows_swap(i, i_max);
			if i != i_max {
				*d = -*d;
			}

			for y in (i + 1)..self.get_height() {
				let f = *self.get(y, j) / *self.get(i, j);
				*self.get_mut(y, j) = T::additive_identity();
				for x in (j + 1)..self.get_width() {
					let val = *self.get(i, x);
					*self.get_mut(y, x) -= val * f;
				}
			}

			i += 1;
			j += 1;
		}
	}

	pub fn to_row_echelon(&mut self) {
		let mut d = T::multiplicative_identity();
		self.to_row_echelon_(&mut d);
	}

	pub fn determinant(&self) -> T {
		let mut d = T::multiplicative_identity();
		let mut n = T::multiplicative_identity();
		let mut mat = self.clone();
		mat.to_row_echelon_(&mut d);

		for i in 0..min(mat.get_height(), mat.get_width()) {
			n *= *mat.get(i, i);
		}
		n / d
	}

	pub fn is_invertible(&self) -> bool {
		self.determinant() != T::additive_identity()
	}

	pub fn get_inverse(&self) -> Self {
		let mut m = Self::new(self.get_height(), self.get_width() * 2);
		for i in 0..self.get_height() {
			for j in 0..self.get_width() {
				*m.get_mut(i, j) = *self.get(i, j);
			}
		}
		for i in 0..self.get_height() {
			*m.get_mut(i, self.get_width() + i) = *self.get(i, i);
		}

		m.to_row_echelon();
		m.submatrix(0, self.get_width(), self.get_height(), self.get_width())
	}

	pub fn rank(&self) -> usize {
		let mut m = self.clone();
		m.to_row_echelon();

		let mut n: usize = 0;
		for i in 0..m.get_height() {
			let mut r = false;
			for j in 0..m.get_width() {
				if *m.get(i, j) != T::additive_identity() {
					r = true;
					break;
				}
			}

			if r {
				n += 1;
			}
		}
		n
	}

	pub fn trace(&self) -> T {
		let max = min(self.get_height(), self.get_width());
		let mut n = T::additive_identity();

		for i in 0..max {
			n += *self.get(i, i);
		}
		n
	}
}

impl<T: Field<T>> Tensor::<T> for Matrix::<T> {
	fn negate(&mut self) {
		for i in &mut self.data {
			*i = -*i;
		}
	}

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

impl<T: Field<T>> std::ops::Neg for Matrix::<T> {
	type Output = Matrix::<T>;

	fn neg(self) -> Self::Output {
		let mut m = self.clone();
		m.negate();
		m
	}
}

impl<T: Field<T>> std::ops::Add<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn add(self, n: T) -> Self::Output {
		let mut m = self.clone();
		m.add_val(&n);
		m
	}
}

impl<T: Field<T>> std::ops::AddAssign<T> for Matrix::<T> {
	fn add_assign(&mut self, n: T) {
		self.add_val(&n);
	}
}

impl<T: Field<T>> std::ops::Sub<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn sub(self, n: T) -> Self::Output {
		let mut m = self.clone();
		m.subtract_val(&n);
		m
	}
}

impl<T: Field<T>> std::ops::SubAssign<T> for Matrix::<T> {
	fn sub_assign(&mut self, n: T) {
		self.subtract_val(&n);
	}
}

impl<T: Field<T>> std::ops::Mul<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn mul(self, n: T) -> Self::Output {
		let mut m = self.clone();
		m.multiply_val(&n);
		m
	}
}

// TODO Multiplication of a matrix by another

impl<T: Field<T>> std::ops::Mul<Vector::<T>> for Matrix::<T> {
	type Output = Vector::<T>;

	fn mul(self, n: Vector::<T>) -> Self::Output {
		// TODO Check that matrix width == vector size

		let mut vec = Vector::<T>::new(self.height);
		for i in 0..vec.get_size() {
			for j in 0..self.get_width() {
				*vec.get_mut(i) += *self.get(i, j) * *n.get(j);
			}
		}
		vec
	}
}

impl<T: Field<T>> std::ops::MulAssign<T> for Matrix::<T> {
	fn mul_assign(&mut self, n: T) {
		self.multiply_val(&n);
	}
}

impl<T: Field<T>> std::ops::Div<T> for Matrix::<T> {
	type Output = Matrix::<T>;

	fn div(self, n: T) -> Self::Output {
		let mut m = self.clone();
		m.divide_val(&n);
		m
	}
}

impl<T: Field<T>> std::ops::DivAssign<T> for Matrix::<T> {
	fn div_assign(&mut self, n: T) {
		self.divide_val(&n);
	}
}

impl<T: Field<T>> Vector::<T> {
	pub fn new(size: usize) -> Self {
		let mut v = Self {
			size: size,
			data: Vec::with_capacity(size),
		};
		v.data.resize(size, T::additive_identity());
		v
	}

	pub fn from_vec(values: Vec::<T>) -> Self {
		let v = Self {
			size: values.len(),
			data: values,
		};
		v
	}

	pub fn get_size(&self) -> usize {
		self.size
	}

	pub fn get(&self, i: usize) -> &T {
		&self.data[i]
	}

	pub fn get_mut(&mut self, i: usize) -> &mut T {
		&mut self.data[i]
	}

	// TODO to_matrix

	pub fn length_squared(&self) -> T {
		let mut n = T::additive_identity();

		for i in 0..self.size {
			let v = self.data[i];
			n += v * v;
		}
		n
	}

	pub fn length(&self) -> T {
		self.length_squared().sqrt()
	}

	pub fn normalize(&mut self) {
		let len = self.length();

		for i in 0..self.size {
			self.data[i] /= len;
		}
	}

	pub fn dot(&self, other: &Vector<T>) -> T {
		let mut n = T::additive_identity();

		// TODO Check other's size
		for i in 0..self.size {
			n += self.data[i] * other.data[i];
		}
		n
	}

	// TODO Cross product
}
