/// This module implements linear algebra utilities.

use crate::Field;
use std::cmp::min;
use std::fmt;
use std::ops::{Index, IndexMut};

/// Trait to implement for tensor objects.
pub trait Tensor<T: Field<T>> {
	/// Negates the tensor.
    fn negate(&mut self);
	/// Adds a value to every elements of the tensor.
    fn add_val(&mut self, n: &T);
	/// Adds another tensor to the tensor.
    fn add_self(&mut self, n: &Self);

	/// Subtracts a value from the tensor.
    fn subtract_val(&mut self, n: &T);
	/// Subtracts another tensor from the tensor.
    fn subtract_self(&mut self, n: &Self);

	/// Multiplies a value to the tensor.
    fn multiply_val(&mut self, n: &T);
	/// Multiplies another tensor to the tensor.
    fn multiply_self(&mut self, n: &Self);

	/// Divides a value from the tensor.
    fn divide_val(&mut self, n: &T);
	/// Divides another tensor from the tensor.
    fn divide_self(&mut self, n: &Self);

	/// Computes the Hadamard product with the given tenor `n`.
	fn hadamard_product(&self, n: &Self) -> Self;
}

/// Structure representing a matrix.
#[derive(Clone)]
pub struct Matrix<T: Field<T>> {
	/// The height of the matrix.
    height: usize,
	/// The width of the matrix.
    width: usize,
	/// The data into the matrix.
    data: Vec<T>,
	/// Whether the vector is transposed or not.
    transposed: bool,
}

/// Structure representing a vector.
#[derive(Clone)]
pub struct Vector<T: Field<T>> {
	/// The data into the vector.
    data: Vec<T>,
}

impl<T: Field<T>> From::<T> for Matrix::<T> {
	fn from(val: T) -> Self {
		let mut vec = Self::new(1, 1);
		*vec.get_mut(0, 0) = val;
		vec
	}
}

impl<T: Field<T>> Matrix::<T> {
	/// Creates a new instance with size `height` and `width`.
    pub fn new(height: usize, width: usize) -> Self {
        assert!(height != 0 && width != 0);

        let mut mat = Self {
            height: height,
            width: width,
            data: Vec::with_capacity(height * width),
            transposed: false,
        };
        mat.data.resize(height * width, T::additive_identity());
        mat
    }

	/// Creates a new instance from the given vector `values` with size `height` and `width`.
    pub fn from_vec(height: usize, width: usize, values: Vec::<T>) -> Self {
        assert!(height != 0 && width != 0);
		assert_eq!(values.len(), height * width);

        let mat = Self {
            height: height,
            width: width,
            data: values,
            transposed: false,
        };
        mat
    }

	/// Creates an identity matrix of size `size` * `size`.
    pub fn identity(size: usize) -> Matrix::<T> {
        let mut mat = Self::new(size, size);

        for i in 0..size {
            *mat.get_mut(i, i) = T::multiplicative_identity();
        }
        mat
    }

	/// Returns the height of the matrix.
    pub fn get_height(&self) -> usize {
        if self.is_transposed() {
            self.width
        } else {
            self.height
        }
    }

	/// Returns the width of the matrix.
    pub fn get_width(&self) -> usize {
        if self.is_transposed() {
            self.height
        } else {
            self.width
        }
    }

	/// Tells whether the matrix is square or not.
    pub fn is_square(&self) -> bool {
        self.height == self.width
    }

	/// Tells whether the matrix is transposed or not.
    pub fn is_transposed(&self) -> bool {
        self.transposed
    }

	/// Returns a linear vector to the matrix's data.
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

	/// Returns a reference to the value at `y`, `x`.
    pub fn get(&self, y: usize, x: usize) -> &T {
        if self.transposed {
            &self.data[x * self.width + y]
        } else {
            &self.data[y * self.width + x]
        }
    }

	/// Returns a mutable reference to the value at `y`, `x`.
    pub fn get_mut(&mut self, y: usize, x: usize) -> &mut T {
        if self.transposed {
            &mut self.data[x * self.width + y]
        } else {
            &mut self.data[y * self.width + x]
        }
    }

	/// Creates a submatrix from position `y`, `x` of size `height` and `width`.
    pub fn submatrix(&self, y: usize, x: usize, height: usize, width: usize) -> Self {
        assert!(y + height <= self.get_height());
        assert!(x + width <= self.get_width());

        let mut m = Self::new(height, width);
        for i in 0..height {
            for j in 0..width {
                *m.get_mut(i, j) = *self.get(y + i, x + j);
            }
        }
        m
    }

	/// Converts the matrix into a vector. If the matrix has more than one column, the behaviour
	/// is undefined.
    pub fn to_vector(&self) -> Vector::<T> {
        assert_eq!(self.get_width(), 1);

        let mut vec = Vector::<T>::new(self.get_height());
        for i in 0..self.get_height() {
            *vec.get_mut(i) = *self.get(i, 0);
        }
        vec
    }

	/// Transposes the matrix.
    pub fn transpose(&mut self) -> &mut Self {
        self.transposed = !self.transposed;
        self
    }

	// TODO Kronecker product

	/// Swaps rows `i` and `j`.
    pub fn rows_swap(&mut self, i: usize, j: usize) {
        assert!(i < self.get_height() && j < self.get_height());

        if i == j {
            return;
        }
        for k in 0..self.get_width() {
            let tmp = *self.get(i, k);
            *self.get_mut(i, k) = *self.get(j, k);
            *self.get_mut(j, k) = tmp;
        }
    }

	/// Columns rows `i` and `j`.
    pub fn columns_swap(&mut self, i: usize, j: usize) {
        assert!(i < self.get_width() && j < self.get_width());

        if i == j {
            return;
        }
        for k in 0..self.get_height() {
            let tmp = *self.get(k, i);
            *self.get_mut(k, i) = *self.get(k, j);
            *self.get_mut(k, j) = tmp;
        }
    }

	/// Passes the matrix to row echelon form. `d` must be a reference to a zero-initialized
	/// variable and will be set with a value used to compute the matrix's determinant.
    fn to_row_echelon_(&mut self, d: &mut T) {
		let mut j = 0;
		let mut r = 0;

		while j < self.get_width() && r < self.get_height() {
            let k = {
                let mut max = r;

                for i in (r + 1)..self.get_height() {
                    if self.get(i, j).abs() > self.get(max, j).abs() {
                        max = i;
                    }
                }

                max
            };

			let val = *self.get(k, j);
			if val != T::additive_identity() {
				for n in j..self.get_width() {
					*self.get_mut(k, n) /= val;
				}

				self.rows_swap(k, r);
				if k != r {
					*d = -*d;
				}

				for i in 0..self.get_height() {
					if i != r {
						let f = *self.get(i, j);
						for n in 0..self.get_width() {
							let v = *self.get(r, n);
							*self.get_mut(i, n) -= v * f;
						}
						*self.get_mut(i, j) = T::additive_identity();
					}
				}

				r += 1;
			}
			j += 1;
		}
    }

	/// Passes the matrix to row echelon form.
    pub fn to_row_echelon(&mut self) {
        let mut d = T::multiplicative_identity();
        self.to_row_echelon_(&mut d);
    }

	/// Computes the determinant of the matrix.
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

	/// Tells whether the matrix is invertible.
    pub fn is_invertible(&self) -> bool {
        self.determinant() != T::additive_identity()
    }

	/// Returns the inverse of the matrix.
    pub fn get_inverse(&self) -> Self {
        let mut m = Self::new(self.get_height(), self.get_width() * 2);
        for i in 0..self.get_height() {
            for j in 0..self.get_width() {
                *m.get_mut(i, j) = *self.get(i, j);
            }
        }
        for i in 0..self.get_height() {
            *m.get_mut(i, self.get_width() + i) = T::multiplicative_identity();
        }

        m.to_row_echelon();
        m.submatrix(0, self.get_width(), self.get_height(), self.get_width())
    }

	/// Returns the rank of the matrix.
    pub fn rank(&self) -> usize {
        let mut m = self.clone();
        m.to_row_echelon();

        let mut n = 0;
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

	/// Tells whether the matrix is full rank.
	pub fn is_full_rank(&self) -> bool {
		self.rank() == min(self.get_height(), self.get_width())
	}

    // TODO Implement for matrices?
	/// Computes the pseudo inverse of the matrix with the given vector `n`.
    pub fn pseudo_inverse(&self, n: &Vector::<T>) -> Vector::<T> {
        let mut transpose = self.clone();
        transpose.transpose();
        (transpose.clone() * self.clone()).get_inverse() * (transpose * n.clone())
    }

	/// Computes the trace of the matrix.
    pub fn trace(&self) -> T {
        let max = min(self.get_height(), self.get_width());
        let mut n = T::additive_identity();

        for i in 0..max {
            n += *self.get(i, i);
        }
        n
    }

    // TODO test
	/// Tells whether the matrix is upper triangular.
    pub fn is_upper_triangular(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        for i in 1..self.get_height() {
            for j in 1..i {
                if !self.get(i, j).epsilon_equal(&T::additive_identity()) {
                    return false;
                }
            }
        }
        true
    }

    // TODO test
	/// Tells whether the matrix is lower triangular.
    pub fn is_lower_triangular(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        for i in 1..self.get_width() {
            for j in 1..i {
                if !self.get(j, i).epsilon_equal(&T::additive_identity()) {
                    return false;
                }
            }
        }
        true
    }

	/// Tells whether the matrix is triangular.
    pub fn is_triangular(&self) -> bool {
        self.is_upper_triangular() || self.is_lower_triangular()
    }

    // TODO Is null
    // TODO Is diagonal
    // TODO Is identity

	// TODO LU decomposition
	// TODO QR decomposition
	// TODO Cholesky decomposition
	// TODO Singular value decomposition
	// TODO Eigenvectors decomposition

	// TODO Forward substitution

	/// TODO doc
	fn back_substitution_(&self, x: &mut Vector::<T>) {
        let mat = self.clone();
		let max = min(mat.get_height(), mat.get_width());

		for i in (0..max).rev() {
			let b = *mat.get(i, mat.get_width() - 1);
			let mut v = T::additive_identity();
			for j in i..(mat.get_width() - 1) {
				v += *mat.get(i, j) * *x.get(j);
			}
			*x.get_mut(i) = (b - v) / *mat.get(i, i);
		}
    }

	/// TODO doc
	pub fn back_substitution(&self) -> Vector::<T> {
        let mut x = Vector::<T>::new(self.get_width() - 1);
		self.back_substitution_(&mut x);
        x
	}

	/// Solves the given system considering the matrix to be augmented.
    pub fn solve(&self) -> Vector::<T> {
        let a = self.submatrix(0, 0, self.get_height(), self.get_width() - 1);
        let b = self.submatrix(0, self.get_width() - 1, self.get_height(), 1).to_vector();
		a.pseudo_inverse(&b)
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
        assert_eq!(self.get_width(), n.get_width());
		assert_eq!(self.get_height(), n.get_height());

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
        assert_eq!(self.get_width(), n.get_width());
		assert_eq!(self.get_height(), n.get_height());

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
        assert_eq!(self.get_width(), n.get_width());
		assert_eq!(self.get_height(), n.get_height());

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
        assert_eq!(self.get_width(), n.get_width());
		assert_eq!(self.get_height(), n.get_height());

        for i in 0..self.data.len() {
            self.data[i] /= n.data[i];
        }
    }

	fn hadamard_product(&self, n: &Self) -> Self {
        assert_eq!(self.get_height(), n.get_height());
		assert_eq!(self.get_width(), n.get_width());

        let mut m = self.clone();
		for i in 0..self.get_height() {
			for j in 0..self.get_width() {
				*m.get_mut(i, j) *= *n.get(i, j);
			}
		}
		m
	}
}

impl<T: Field<T>> std::ops::Neg for Matrix::<T> {
    type Output = Matrix::<T>;

    fn neg(mut self) -> Self::Output {
        self.negate();
        self
    }
}

impl<T: Field<T>> std::ops::Add<T> for Matrix::<T> {
    type Output = Matrix::<T>;

    fn add(mut self, n: T) -> Self::Output {
        self.add_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::AddAssign<T> for Matrix::<T> {
    fn add_assign(&mut self, n: T) {
        self.add_val(&n);
    }
}

impl<T: Field<T>> std::ops::Add<Matrix::<T>> for Matrix::<T> {
    type Output = Matrix::<T>;

    fn add(mut self, n: Matrix::<T>) -> Self::Output {
        self.add_self(&n);
        self
    }
}

impl<T: Field<T>> std::ops::AddAssign<Matrix::<T>> for Matrix::<T> {
    fn add_assign(&mut self, n: Matrix::<T>) {
        self.add_self(&n);
    }
}

impl<T: Field<T>> std::ops::Sub<T> for Matrix::<T> {
    type Output = Matrix::<T>;

    fn sub(mut self, n: T) -> Self::Output {
        self.subtract_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::SubAssign<T> for Matrix::<T> {
    fn sub_assign(&mut self, n: T) {
        self.subtract_val(&n);
    }
}

impl<T: Field<T>> std::ops::Sub<Matrix::<T>> for Matrix::<T> {
    type Output = Matrix::<T>;

    fn sub(mut self, n: Matrix::<T>) -> Self::Output {
        self.subtract_self(&n);
        self
    }
}

impl<T: Field<T>> std::ops::SubAssign<Matrix::<T>> for Matrix::<T> {
    fn sub_assign(&mut self, n: Matrix::<T>) {
        self.subtract_self(&n);
    }
}

impl<T: Field<T>> std::ops::Mul<T> for Matrix::<T> {
    type Output = Matrix::<T>;

    fn mul(mut self, n: T) -> Self::Output {
        self.multiply_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::Mul<Matrix::<T>> for Matrix::<T> {
    type Output = Matrix::<T>;

    fn mul(self, n: Matrix::<T>) -> Self::Output {
        assert_eq!(self.width, n.height);

        let mut mat = Self::new(self.get_height(), n.get_width());
        for i in 0..mat.get_height() {
            for j in 0..mat.get_width() {
                let mut v = T::additive_identity();

                for k in 0..self.get_width() {
                    v = (*self.get(i, k)).mul_add(n.get(k, j), &v);
                }
                *mat.get_mut(i, j) = v;
            }
        }
        mat
    }
}

impl<T: Field<T>> std::ops::Mul<Vector::<T>> for Matrix::<T> {
    type Output = Vector::<T>;

    fn mul(self, n: Vector::<T>) -> Self::Output {
        assert_eq!(self.width, n.get_size());

        let mut vec = Vector::<T>::new(self.get_height());
        for i in 0..vec.get_size() {
            for j in 0..self.get_width() {
                let v = vec.get_mut(i);
                *v = (*self.get(i, j)).mul_add(n.get(j), v);
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

    fn div(mut self, n: T) -> Self::Output {
        self.divide_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::DivAssign<T> for Matrix::<T> {
    fn div_assign(&mut self, n: T) {
        self.divide_val(&n);
    }
}

impl<T: Field<T>> std::fmt::Display for Matrix::<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.get_height() {
            for j in 0..self.get_width() {
                let _ = write!(f, "{}", *self.get(i, j));
                if j < self.get_width() - 1 {
                    let _ = write!(f, " ");
                }
            }
            let _ = write!(f, "\n");
        }
        let _ = write!(f, "");
		Ok(())
    }
}

impl<T: Field<T>> From::<T> for Vector::<T> {
	fn from(val: T) -> Self {
		let mut vec = Self::new(1);
		*vec.x_mut() = val;
		vec
	}
}

impl<T: Field<T>> Vector::<T> {
	/// Creates a new instance of size `size`.
    pub fn new(size: usize) -> Self {
    	assert!(size > 0);
        let mut v = Self {
            data: Vec::with_capacity(size),
        };
        v.data.resize(size, T::additive_identity());
        v
    }

	/// Creates a new instance with values in `values`.
    pub fn from_vec(values: Vec::<T>) -> Self {
        let v = Self {
            data: values,
        };
        v
    }

	/// Returns the size of the vector.
    pub fn get_size(&self) -> usize {
        self.data.len()
    }

	/// Returns a vec containing the data of the vector.
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

	/// Returns a reference to the `i`th element of the vector.
    pub fn get(&self, i: usize) -> &T {
        &self.data[i]
    }

	/// Returns a mutable reference to the `i`th element of the vector.
    pub fn get_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }

	/// Returns a reference to the x element of the vector.
    pub fn x(&self) -> &T {
        self.get(0)
    }

	/// Returns a mutable reference to the x element of the vector.
    pub fn x_mut(&mut self) -> &mut T {
        self.get_mut(0)
    }

	/// Returns a reference to the y element of the vector.
    pub fn y(&self) -> &T {
        self.get(1)
    }

	/// Returns a mutable reference to the y element of the vector.
    pub fn y_mut(&mut self) -> &mut T {
        self.get_mut(1)
    }

	/// Returns a reference to the z element of the vector.
    pub fn z(&self) -> &T {
        self.get(2)
    }

	/// Returns a mutable reference to the z element of the vector.
    pub fn z_mut(&mut self) -> &mut T {
        self.get_mut(2)
    }

	/// Returns a reference to the w element of the vector.
    pub fn w(&self) -> &T {
        self.get(3)
    }

	/// Returns a mutable reference to the w element of the vector.
    pub fn w_mut(&mut self) -> &mut T {
        self.get_mut(3)
    }

	/// Returns a matrix equivalent to the current vector.
	pub fn to_matrix(&self) -> Matrix::<T> {
		let size = self.get_size();
		let mut mat = Matrix::new(size, 1);
		for i in 0..size {
			*mat.get_mut(i, 0) = *self.get(i);
		}
		mat
	}

	// TODO 1-norm, p-norm and inf-norm

	/// Returns the squared euclidean length of the vector.
    pub fn length_squared(&self) -> T {
        let mut n = T::additive_identity();

        for i in 0..self.data.len() {
            let v = self.data[i];
            n = v.mul_add(&v, &n);
        }
        n
    }

	/// Returns the length of the vector.
    pub fn length(&self) -> T {
        self.length_squared().sqrt()
    }

	/// Normalizes the vector to the given length `length`.
    pub fn normalize(&mut self, length: T) -> &mut Self {
        let len = self.length();

        for i in 0..self.data.len() {
            self.data[i] = (self.data[i] * length) / len;
        }
        self
    }

	/// Computes the dot product with the other vector `other`.
    pub fn dot(&self, other: &Vector<T>) -> T {
		assert_eq!(self.get_size(), other.get_size());

        let mut n = T::additive_identity();
        for i in 0..self.data.len() {
            n = self.data[i].mul_add(&other.data[i], &n);
        }
        n
    }

	/// Returns the cosine of the angle between the current vector and `other`. If one or both
	/// vectors have a size of zero, the behaviour is undefined.
	pub fn cosine(&self, other: &Vector<T>) -> T {
		self.dot(other) / self.length() * other.length()
	}

	/// Computes the cross product between the current vector and `other`. If the vector isn't
	/// 3-dimensional, the behaviour is undefined.
    pub fn cross_product(&self, other: &Vector<T>) -> Self {
		assert_eq!(self.get_size(), 3);
		assert_eq!(other.get_size(), 3);

        Self::from_vec(vec!{
            self.get(1).mul_add(other.get(2), &-(*self.get(2) * *other.get(1))),
            self.get(2).mul_add(other.get(0), &-(*self.get(0) * *other.get(2))),
            self.get(0).mul_add(other.get(1), &-(*self.get(1) * *other.get(0))),
        })
    }

	// TODO Unit tests
	/// Computes the outer product of the current vector and `other`.
    pub fn outer_product(&self, other: &Vector<T>) -> Matrix<T> {
    	let mut mat = Matrix::new(self.get_size(), other.get_size());
    	for i in 0..self.get_size() {
    		for j in 0..self.get_size() {
    			*mat.get_mut(i, j) = *self.get(i) * *other.get(j);
    		}
    	}
    	mat
    }

	// TODO Unit tests
    /// Executes the given function `f` for each elements in the vector.
    /// The first argument of the closure is the value and the second argument is the index in the
    /// vector.
    /// The return value is placed back into the vector at the given index.
    pub fn for_each<F: FnMut(T, usize) -> T>(&mut self, mut f: F) {
    	for i in 0..self.data.len() {
    		let val = self.data[i];
    		self.data[i] = f(val, i);
    	}
    }
}

impl<T: Field<T>> Tensor::<T> for Vector::<T> {
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
        assert_eq!(self.get_size(), n.get_size());

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
        assert_eq!(self.get_size(), n.get_size());

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
        assert_eq!(self.get_size(), n.get_size());

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
        assert_eq!(self.get_size(), n.get_size());

        for i in 0..self.data.len() {
            self.data[i] /= n.data[i];
        }
    }

	fn hadamard_product(&self, n: &Self) -> Self {
        assert_eq!(self.get_size(), n.get_size());

        let mut v = self.clone();
		for i in 0..self.get_size() {
			*v.get_mut(i) *= *n.get(i);
		}
		v
	}
}

impl<T: Field<T>> Index<usize> for Vector::<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
		self.get(i)
    }
}

impl<T: Field<T>> IndexMut<usize> for Vector::<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
		self.get_mut(i)
    }
}

impl<T: Field<T>> std::ops::Neg for Vector::<T> {
    type Output = Vector::<T>;

    fn neg(mut self) -> Self::Output {
        self.negate();
        self
    }
}

impl<T: Field<T>> std::ops::Add<T> for Vector::<T> {
    type Output = Vector::<T>;

    fn add(mut self, n: T) -> Self::Output {
        self.add_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::Add<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn add(mut self, n: Vector::<T>) -> Self::Output {
        self.add_self(&n);
        self
    }
}

impl<T: Field<T>> std::ops::AddAssign<T> for Vector::<T> {
    fn add_assign(&mut self, n: T) {
        self.add_val(&n);
    }
}

impl<T: Field<T>> std::ops::AddAssign<Vector::<T>> for Vector::<T> {
    fn add_assign(&mut self, n: Vector::<T>) {
        self.add_self(&n);
    }
}

impl<T: Field<T>> std::ops::Sub<T> for Vector::<T> {
    type Output = Vector::<T>;

    fn sub(mut self, n: T) -> Self::Output {
        self.subtract_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::Sub<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn sub(mut self, n: Vector::<T>) -> Self::Output {
        self.subtract_self(&n);
        self
    }
}

impl<T: Field<T>> std::ops::SubAssign<T> for Vector::<T> {
    fn sub_assign(&mut self, n: T) {
        self.subtract_val(&n);
    }
}

impl<T: Field<T>> std::ops::SubAssign<Vector::<T>> for Vector::<T> {
    fn sub_assign(&mut self, n: Vector::<T>) {
        self.subtract_self(&n);
    }
}

impl<T: Field<T>> std::ops::Mul<T> for Vector::<T> {
    type Output = Vector::<T>;

    fn mul(mut self, n: T) -> Self::Output {
        self.multiply_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::Mul<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn mul(mut self, n: Vector::<T>) -> Self::Output {
        self.multiply_self(&n);
        self
    }
}

impl<T: Field<T>> std::ops::MulAssign<T> for Vector::<T> {
    fn mul_assign(&mut self, n: T) {
        self.multiply_val(&n);
    }
}

impl<T: Field<T>> std::ops::MulAssign<Vector::<T>> for Vector::<T> {
    fn mul_assign(&mut self, n: Vector::<T>) {
        self.multiply_self(&n);
    }
}

impl<T: Field<T>> std::ops::Div<T> for Vector::<T> {
    type Output = Vector::<T>;

    fn div(mut self, n: T) -> Self::Output {
        self.divide_val(&n);
        self
    }
}

impl<T: Field<T>> std::ops::Div<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn div(mut self, n: Vector::<T>) -> Self::Output {
        self.divide_self(&n);
        self
    }
}

impl<T: Field<T>> std::ops::DivAssign<T> for Vector::<T> {
    fn div_assign(&mut self, n: T) {
        self.divide_val(&n);
    }
}

impl<T: Field<T>> std::ops::DivAssign<Vector::<T>> for Vector::<T> {
    fn div_assign(&mut self, n: Vector::<T>) {
        self.divide_self(&n);
    }
}

impl<T: Field<T>> std::fmt::Display for Vector::<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.get_size() {
            let _ = write!(f, "{}", *self.get(i));
            if i < self.get_size() - 1 {
                let _ = write!(f, " ");
            }
        }
        let _ = write!(f, "");
		Ok(())
    }
}

#[cfg(test)]
mod tests {
	use super::*;

	macro_rules! assert_eq_delta {
		($n0:expr, $n1:expr) => {
			let r = ($n0).epsilon_equal(&($n1));
			if !r {
				eprintln!("Delta assert fail: {} != {}", $n0, $n1);
			}
			assert!(r);
		}
	}

	#[test]
	fn test_mat_add() {
		let mut mat = Matrix::<f64>::new(3, 3);
		mat += 1.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq_delta!(*mat.get(i, j), 1. as f64);
			}
		}
	}

	#[test]
	fn test_mat_sub() {
		let mut mat = Matrix::<f64>::new(3, 3);
		mat -= 1.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq_delta!(*mat.get(i, j), -1. as f64);
			}
		}
	}

	#[test]
	fn test_mat_mul() {
		let mut mat = Matrix::<f64>::new(3, 3);
		for i in 0..3 {
			for j in 0..3 {
				*mat.get_mut(i, j) = (i * 3 + j) as f64;
			}
		}
		mat *= 2.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq_delta!(*mat.get(i, j), (i * 3 + j) as f64 * 2.);
			}
		}
	}

	#[test]
	fn test_mat_mul_mat0() {
		let mat0 = Matrix::<f64>::identity(3);
		let mat1 = Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,
		});
		let mat2 = mat0 * mat1;

		assert_eq_delta!(*mat2.get(0, 0), 0. as f64);
		assert_eq_delta!(*mat2.get(0, 1), 1. as f64);
		assert_eq_delta!(*mat2.get(0, 2), 2. as f64);
		assert_eq_delta!(*mat2.get(1, 0), 3. as f64);
		assert_eq_delta!(*mat2.get(1, 1), 4. as f64);
		assert_eq_delta!(*mat2.get(1, 2), 5. as f64);
		assert_eq_delta!(*mat2.get(2, 0), 6. as f64);
		assert_eq_delta!(*mat2.get(2, 1), 7. as f64);
		assert_eq_delta!(*mat2.get(2, 2), 8. as f64);
	}

	#[test]
	fn test_mat_mul_mat1() {
		let mat0 = Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,
		});
		let mat2 = mat0.clone() * mat0;

		assert_eq_delta!(*mat2.get(0, 0), 15. as f64);
		assert_eq_delta!(*mat2.get(0, 1), 18. as f64);
		assert_eq_delta!(*mat2.get(0, 2), 21. as f64);
		assert_eq_delta!(*mat2.get(1, 0), 42. as f64);
		assert_eq_delta!(*mat2.get(1, 1), 54. as f64);
		assert_eq_delta!(*mat2.get(1, 2), 66. as f64);
		assert_eq_delta!(*mat2.get(2, 0), 69. as f64);
		assert_eq_delta!(*mat2.get(2, 1), 90. as f64);
		assert_eq_delta!(*mat2.get(2, 2), 111. as f64);
	}

	#[test]
	fn test_mat_mul_vec0() {
		let mat = Matrix::<f64>::identity(3);
		let vec0 = Vector::<f64>::from_vec(vec!{
			0., 1., 2.,
		});
		let vec1 = mat * vec0;

		assert_eq_delta!(*vec1.get(0), 0. as f64);
		assert_eq_delta!(*vec1.get(1), 1. as f64);
		assert_eq_delta!(*vec1.get(2), 2. as f64);
	}

	#[test]
	fn test_mat_mul_vec1() {
		let mat = Matrix::<f64>::identity(3) * 2.;
		let vec0 = Vector::<f64>::from_vec(vec!{
			0., 1., 2.,
		});
		let vec1 = mat * vec0;

		assert_eq_delta!(*vec1.get(0), 0. as f64);
		assert_eq_delta!(*vec1.get(1), 2. as f64);
		assert_eq_delta!(*vec1.get(2), 4. as f64);
	}

	#[test]
	fn test_mat_mul_vec2() {
		let mat = Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 0.,
			1., 0., 1.,
			0., 1., 0.,
		});
		let vec0 = Vector::<f64>::from_vec(vec!{
			0., 1., 2.,
		});
		let vec1 = mat * vec0;

		assert_eq_delta!(*vec1.get(0), 1. as f64);
		assert_eq_delta!(*vec1.get(1), 2. as f64);
		assert_eq_delta!(*vec1.get(2), 1. as f64);
	}

	#[test]
	fn test_mat_div() {
		let mut mat = Matrix::<f64>::new(3, 3);
		for i in 0..3 {
			for j in 0..3 {
				*mat.get_mut(i, j) = (i * 3 + j) as f64;
			}
		}
		mat /= 2.;
		for i in 0..3 {
			for j in 0..3 {
				assert_eq_delta!(*mat.get(i, j), (i * 3 + j) as f64 / 2.);
			}
		}
	}

	#[test]
	fn test_mat_transpose() {
		let mut mat = Matrix::<f64>::from_vec(4, 3, vec!{
			1., 0., 0.,
			2., 0., 3.,
			0., 0., 0.,
			0., 0., 0.,
		});

		assert!(!mat.is_transposed());
		assert_eq!(mat.get_height(), 4);
		assert_eq!(mat.get_width(), 3);
		assert_eq_delta!(*mat.get(0, 0), 1.);
		assert_eq_delta!(*mat.get(1, 0), 2.);
		assert_eq_delta!(*mat.get(1, 2), 3.);
		assert_eq_delta!(*mat.get(0, 1), 0.);
		assert_eq_delta!(*mat.get(2, 1), 0.);

		mat.transpose();

		assert!(mat.is_transposed());
		assert_eq!(mat.get_height(), 3);
		assert_eq!(mat.get_width(), 4);
		assert_eq_delta!(*mat.get(0, 0), 1.);
		assert_eq_delta!(*mat.get(0, 1), 2.);
		assert_eq_delta!(*mat.get(2, 1), 3.);
		assert_eq_delta!(*mat.get(1, 0), 0.);
		assert_eq_delta!(*mat.get(1, 2), 0.);
	}

	// TODO Hadamard product
	// TODO Kronecker product

	#[test]
	fn test_mat_rows_swap0() {
		let mut mat = Matrix::<f64>::from_vec(3, 3, vec!{
            0., 1., 2.,
            3., 4., 5.,
            6., 7., 8.,
        });
		mat.rows_swap(0, 1);

        assert_eq_delta!(*mat.get(0, 0), 3.);
        assert_eq_delta!(*mat.get(0, 1), 4.);
        assert_eq_delta!(*mat.get(0, 2), 5.);
        assert_eq_delta!(*mat.get(1, 0), 0.);
        assert_eq_delta!(*mat.get(1, 1), 1.);
        assert_eq_delta!(*mat.get(1, 2), 2.);
        assert_eq_delta!(*mat.get(2, 0), 6.);
        assert_eq_delta!(*mat.get(2, 1), 7.);
        assert_eq_delta!(*mat.get(2, 2), 8.);
	}

	#[test]
	fn test_mat_rows_swap1() {
		let mut mat = Matrix::<f64>::from_vec(3, 3, vec!{
            0., 1., 2.,
            3., 4., 5.,
            6., 7., 8.,
        });
		mat.rows_swap(0, 2);

        assert_eq_delta!(*mat.get(0, 0), 6.);
        assert_eq_delta!(*mat.get(0, 1), 7.);
        assert_eq_delta!(*mat.get(0, 2), 8.);
        assert_eq_delta!(*mat.get(1, 0), 3.);
        assert_eq_delta!(*mat.get(1, 1), 4.);
        assert_eq_delta!(*mat.get(1, 2), 5.);
        assert_eq_delta!(*mat.get(2, 0), 0.);
        assert_eq_delta!(*mat.get(2, 1), 1.);
        assert_eq_delta!(*mat.get(2, 2), 2.);
	}

	#[test]
	fn test_mat_row_echelon0() {
		let mut mat = Matrix::<f64>::new(3, 3);
		mat.to_row_echelon();

		for i in 0..mat.get_height() {
			for j in 0..mat.get_width() {
				assert_eq_delta!(*mat.get(i, j), 0.);
			}
		}
	}

	#[test]
	fn test_mat_row_echelon1() {
		let mut mat = Matrix::<f64>::identity(3);
		mat.to_row_echelon();

		for i in 0..mat.get_height() {
			for j in 0..mat.get_width() {
				assert_eq_delta!(*mat.get(i, j), if i == j { 1. } else { 0. });
			}
		}
	}

	#[test]
	fn test_mat_row_echelon2() {
		let mut mat = Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,
		});
		mat.to_row_echelon();

		assert_eq_delta!(*mat.get(0, 0), 1.);
		assert_eq_delta!(*mat.get(0, 1), 0.);
		assert_eq_delta!(*mat.get(0, 2), -1.);
		assert_eq_delta!(*mat.get(1, 0), 0.);
		assert_eq_delta!(*mat.get(1, 1), 1.);
		assert_eq_delta!(*mat.get(1, 2), 2.);
		assert_eq_delta!(*mat.get(2, 0), 0.);
		assert_eq_delta!(*mat.get(2, 1), 0.);
		assert_eq_delta!(*mat.get(2, 2), 0.);
	}

	#[test]
	fn test_mat_row_echelon3() {
		let mut mat = Matrix::<f64>::from_vec(3, 3, vec!{
			2., -1., 0.,
			-1., 2., -1.,
			0., -1., 2.,
		});
		mat.to_row_echelon();

		assert_eq_delta!(*mat.get(0, 0), 1.);
		assert_eq_delta!(*mat.get(0, 1), 0.);
		assert_eq_delta!(*mat.get(0, 2), 0.);
		assert_eq_delta!(*mat.get(1, 0), 0.);
		assert_eq_delta!(*mat.get(1, 1), 1.);
		assert_eq_delta!(*mat.get(1, 2), 0.);
		assert_eq_delta!(*mat.get(2, 0), 0.);
		assert_eq_delta!(*mat.get(2, 1), 0.);
		assert_eq_delta!(*mat.get(2, 2), 1.);
	}

	#[test]
	fn test_mat_row_echelon4() {
		let mut mat = Matrix::<f64>::from_vec(3, 6, vec!{
			0., 1., 2., 1., 0., 0.,
			3., 4., 5., 0., 1., 0.,
			6., 7., 8., 0., 0., 1.,
		});
		mat.to_row_echelon();

		assert_eq_delta!(*mat.get(0, 0), 1.);
		assert_eq_delta!(*mat.get(0, 1), 0.);
		assert_eq_delta!(*mat.get(0, 2), -1.);
		assert_eq_delta!(*mat.get(0, 3), 0.);
		assert_eq_delta!(*mat.get(0, 4), -2. - (1. / 3.));
		assert_eq_delta!(*mat.get(0, 5), 1. + (1. / 3.));

		assert_eq_delta!(*mat.get(1, 0), 0.);
		assert_eq_delta!(*mat.get(1, 1), 1.);
		assert_eq_delta!(*mat.get(1, 2), 2.);
		assert_eq_delta!(*mat.get(1, 3), 0.);
		assert_eq_delta!(*mat.get(1, 4), 2.);
		assert_eq_delta!(*mat.get(1, 5), -1.);

		assert_eq_delta!(*mat.get(2, 0), 0.);
		assert_eq_delta!(*mat.get(2, 1), 0.);
		assert_eq_delta!(*mat.get(2, 2), 0.);
		assert_eq_delta!(*mat.get(2, 3), 1.);
		assert_eq_delta!(*mat.get(2, 4), -2.);
		assert_eq_delta!(*mat.get(2, 5), 1.);
	}

	#[test]
	fn test_mat_row_echelon5() {
		let mut mat = Matrix::<f64>::new(10, 10);
        for i in 0..mat.get_height() {
            for j in 0..mat.get_width() {
                *mat.get_mut(i, j) = 1.;
            }
        }
		mat.to_row_echelon();

        for i in 0..mat.get_height() {
            for j in 0..mat.get_width() {
                assert_eq_delta!(*mat.get(i, j), if i == 0 { 1. } else { 0. });
            }
        }
	}

	#[test]
	fn test_mat_determinant0() {
		let mat = Matrix::<f64>::new(3, 3);
		assert_eq_delta!(mat.determinant(), 0 as f64);
	}

	#[test]
	fn test_mat_determinant1() {
		let mat = Matrix::<f64>::identity(3);
		assert_eq_delta!(mat.determinant(), 1 as f64);
	}

	#[test]
	fn test_mat_determinant2() {
		let mat = Matrix::<f64>::from_vec(3, 3, vec!{
			-2., 2., -3.,
			-1., 1., 3.,
			2., 0., -1.,
		});

		assert_eq_delta!(mat.determinant(), 18 as f64);
	}

	#[test]
	fn test_mat_determinant3() {
		let mat = Matrix::<f64>::from_vec(3, 3, vec!{
			-2., 2., -3.,
			0., 2., -4.,
			0., 0., 4.5,
		});

		assert_eq_delta!(mat.determinant(), -18 as f64);
	}

	#[test]
	fn test_mat_inverse0() {
		let mat = Matrix::<f64>::identity(3);
		let inverse = mat.get_inverse();

		for i in 0..inverse.get_height() {
			for j in 0..inverse.get_width() {
				assert_eq_delta!(*inverse.get(i, j), if i == j { 1. } else { 0. });
			}
		}
	}

	#[test]
	fn test_mat_inverse1() {
		let mat = Matrix::<f64>::from_vec(3, 3, vec!{
			2., -1., 0.,
			-1., 2., -1.,
			0., -1., 2.,
		});
		let inverse = mat.get_inverse();

		assert_eq_delta!(*inverse.get(0, 0), 0.75);
		assert_eq_delta!(*inverse.get(0, 1), 0.5);
		assert_eq_delta!(*inverse.get(0, 2), 0.25);
		assert_eq_delta!(*inverse.get(1, 0), 0.5);
		assert_eq_delta!(*inverse.get(1, 1), 1.);
		assert_eq_delta!(*inverse.get(1, 2), 0.5);
		assert_eq_delta!(*inverse.get(2, 0), 0.25);
		assert_eq_delta!(*inverse.get(2, 1), 0.5);
		assert_eq_delta!(*inverse.get(2, 2), 0.75);
	}

	#[test]
	fn test_mat_rank0() {
		let mat = Matrix::<f64>::new(3, 3);
		assert_eq!(mat.rank(), 0);
	}

	#[test]
	fn test_mat_rank1() {
		let mat = Matrix::<f64>::identity(3);
		assert_eq!(mat.rank(), 3);
	}

	#[test]
	fn test_mat_rank2() {
		let mat = Matrix::<f64>::from_vec(3, 3, vec!{
			0., 1., 2.,
			3., 4., 5.,
			6., 7., 8.,
		});
		assert_eq!(mat.rank(), 2);
	}

	#[test]
	fn test_mat_trace0() {
		let mut mat = Matrix::<f64>::new(4, 3);
		mat += 1.;
		assert_eq_delta!(mat.trace(), 3. as f64);
	}

	fn test_system(system: &Matrix::<f64>) {
		let r = system.solve();
		let r_size = r.get_size();
		assert_eq!(r_size, system.get_width() - 1);

		let a = system.submatrix(0, 0, system.get_height(), r_size) * r.clone(); // TODO rm clone
		assert_eq!(a.get_size(), system.get_height());
        for i in 0..a.get_size() {
            assert_eq_delta!(*a.get(i), *system.get(i, system.get_width() - 1));
        }
	}

	#[test]
	fn test_mat_solve0() {
		let mat = Matrix::<f64>::from_vec(3, 4, vec!{
			1., 0., 0., 1.,
			0., 1., 0., 1.,
			0., 0., 1., 1.,
		});
		test_system(&mat);
	}

	#[test]
	fn test_mat_solve1() {
		let mat = Matrix::<f64>::from_vec(2, 3, vec!{
			4., 2., -1.,
			3., -1., 2.,
		});
		test_system(&mat);
	}

	#[test]
	fn test_mat_solve2() {
		let mat = Matrix::<f64>::from_vec(3, 4, vec!{
			3., 2., -1., 1.,
			2., -2., 4., -2.,
			-1., 0.5, -1., 0.,
		});
		test_system(&mat);
	}

	/*#[test]
	fn test_mat_solve3() {
		let mat = Matrix::<f64>::from_vec(2, 4, vec!{
			1., 1., 1., 1.,
			1., 1., 2., 3.,
		});
		test_system(&mat);
	}*/

	#[test]
	fn test_mat_solve4() {
		let mut mat = Matrix::<f64>::new(100, 101);
        for i in 0..mat.get_height() {
            for j in 0..mat.get_width() {
                *mat.get_mut(i, j) = 1.;
            }
        }
		test_system(&mat);
	}

	#[test]
	fn test_vec_length0() {
		let vec = Vector::<f64>::from_vec(vec!{1., 0., 0.});
		assert_eq_delta!(vec.length(), 1. as f64);
	}

	#[test]
	fn test_vec_length1() {
		let vec = Vector::<f64>::from_vec(vec!{1., 1., 1.});
		assert_eq_delta!(vec.length(), (3. as f64).sqrt());
	}

	#[test]
	fn test_vec_normalize0() {
		let mut vec = Vector::<f64>::from_vec(vec!{1., 1., 1.});
		vec.normalize(1.);
		assert_eq_delta!(vec.length(), 1. as f64);
	}

	#[test]
	fn test_vec_dot0() {
		let vec0 = Vector::<f64>::from_vec(vec!{1., 0., 0.});
		let vec1 = Vector::<f64>::from_vec(vec!{0., 1., 0.});
		assert_eq_delta!(vec0.dot(&vec1), 0. as f64);
	}

	// TODO Test cosine

	#[test]
	fn test_vec_cross_product0() {
		let vec0 = Vector::<f64>::from_vec(vec!{1., 0., 0.});
		let vec1 = Vector::<f64>::from_vec(vec!{0., 1., 0.});
		let vec2 = vec0.cross_product(&vec1);
		assert_eq_delta!(*vec2.get(0), 0. as f64);
		assert_eq_delta!(*vec2.get(1), 0. as f64);
		assert_eq_delta!(*vec2.get(2), 1. as f64);
	}

	#[test]
	fn test_vec_cross_product1() {
		let vec0 = Vector::<f64>::from_vec(vec!{1., 0.5, 2.});
		let vec1 = Vector::<f64>::from_vec(vec!{0.8, 1., 0.});
		let vec2 = vec0.cross_product(&vec1);
		assert_eq_delta!(*vec2.get(0), -2. as f64);
		assert_eq_delta!(*vec2.get(1), 1.6 as f64);
		assert_eq_delta!(*vec2.get(2), 0.6 as f64);
	}
}
