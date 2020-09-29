use crate::Field;
use std::cmp::min;
use std::fmt;

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

    pub fn get_data(&self) -> &Vec<T> {
        &self.data
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

    pub fn transpose(&mut self) -> &mut Self {
        self.transposed = !self.transposed;
        self
    }

    pub fn rows_swap(&mut self, i: usize, j: usize) {
        assert!(i < self.get_height());
        assert!(j < self.get_height());

        if i == j {
            return;
        }
        for k in 0..self.get_width() {
            let tmp = *self.get(i, k);
            *self.get_mut(i, k) = *self.get(j, k);
            *self.get_mut(j, k) = tmp;
        }
    }

    pub fn columns_swap(&mut self, i: usize, j: usize) {
        assert!(i < self.get_width());
        assert!(j < self.get_width());

        if i == j {
            return;
        }
        for k in 0..self.get_height() {
            let tmp = *self.get(k, i);
            *self.get_mut(k, i) = *self.get(k, j);
            *self.get_mut(k, j) = tmp;
        }
    }

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
            *m.get_mut(i, self.get_width() + i) = T::multiplicative_identity();
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

    // TODO Implement for matrices?
    pub fn pseudo_inverse(&self, n: &Vector::<T>) -> Vector::<T> {
        let mut transpose = self.clone();
        transpose.transpose();
        (transpose.clone() * self.clone()).get_inverse() * (transpose * n.clone())
    }

    pub fn trace(&self) -> T {
        let max = min(self.get_height(), self.get_width());
        let mut n = T::additive_identity();

        for i in 0..max {
            n += *self.get(i, i);
        }
        n
    }

	// TODO Is triangular
	// TODO Is upper triangular
	// TODO Is lower triangular
	// TODO LU decomposition
	// TODO QR decomposition
	// TODO Forward substitution

    // TODO Clean
	fn back_substitution_(&self, x: &mut Vector::<T>) {
        let mut mat = self.clone();
		let max = min(mat.get_height(), mat.get_width());
        let mut swaps = Vec::<(usize, usize)>::new();
		let mut solved = Vec::<bool>::new();
		solved.resize(mat.get_width() - 1, false);

        println!("begin ->\n{}", mat);

		for i in (0..max).rev() {
            if *mat.get(i, i) == T::additive_identity() {
                let mut pivot = 0;
                for k in i..(mat.get_width() - 1) {
                    if *mat.get(i, k) != T::additive_identity() && !solved[k] {
                        pivot = k;
                        break;
                    }
                }
                if pivot != i {
                    println!("swap -> {} {}", i, pivot);
                    swaps.push((i, pivot));
                    mat.columns_swap(i, pivot);
                    let n = *x.get(i);
                    *x.get_mut(i) = *x.get(pivot);
                    *x.get_mut(pivot) = n;
                    println!("swap result ->\n{}", mat);
                }
            }

			let b = *mat.get(i, mat.get_width() - 1);
			let mut v = T::additive_identity();
			for j in i..(mat.get_width() - 1) {
				v += *mat.get(i, j) * *x.get(j);
			}
            println!("row {} sum: {}", i, v);

			*x.get_mut(i) = (b - v) / *mat.get(i, i);
            println!("x{} = {}", i, *x.get_mut(i));
			solved[i] = true;
		}

        println!("before unswap -> {}", x);
        for i in (0..swaps.len()).rev() {
            let (s0, s1) = swaps[i];
            let n = *x.get(s0);
            *x.get_mut(s0) = *x.get(s1);
            *x.get_mut(s1) = n;
        }
        println!("after unswap -> {}", x);
		// TODO Assert that every field is solved
    }

	pub fn back_substitution(&self) -> Vector::<T> {
        let mut x = Vector::<T>::new(self.get_width() - 1);
		self.back_substitution_(&mut x);
        x
	}

    pub fn solve(&self) -> Vector::<T> {
        let mut mat = self.clone();
        mat.to_row_echelon();

        // TODO Handle overdetermined systems
        let mut x = Vector::<T>::new(mat.get_width() - 1);
        if mat.get_height() < mat.get_width() - 1 {
            let mut v = Vector::<T>::new(mat.get_width() - mat.get_height());
            for i in 0..v.get_size() {
                *v.get_mut(i) = *mat.get(mat.get_height() - 1, mat.get_height() - 1 + i);
            }

            let v_len2 = v.clone().length_squared();
            let u = v.clone() / v_len2 * *mat.get(mat.get_height() - 1, mat.get_width() - 1);
            for i in 0..u.get_size() {
                *mat.get_mut(mat.get_height() - 1, mat.get_height() - 1 + i) = *u.get(i);
            }

            mat.back_substitution_(&mut x);
        } else {
            mat.back_substitution_(&mut x);
        }
        x
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

impl<T: Field<T>> std::ops::Mul<Matrix::<T>> for Matrix::<T> {
    type Output = Matrix::<T>;

    fn mul(self, n: Matrix::<T>) -> Self::Output {
        // TODO Check that self.width == n.height

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
        // TODO Check that matrix width == vector size

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

impl<T: Field<T>> std::fmt::Display for Matrix::<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.get_height() {
            for j in 0..self.get_width() {
                write!(f, "{}", *self.get(i, j));
                if j < self.get_width() - 1 {
                    write!(f, " ");
                }
            }
            write!(f, "\n");
        }
        write!(f, "")
    }
}

impl<T: Field<T>> Vector::<T> {
    pub fn new(size: usize) -> Self {
        let mut v = Self {
            data: Vec::with_capacity(size),
        };
        v.data.resize(size, T::additive_identity());
        v
    }

    pub fn from_vec(values: Vec::<T>) -> Self {
        let v = Self {
            data: values,
        };
        v
    }

    pub fn get_size(&self) -> usize {
        self.data.len()
    }

    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn get(&self, i: usize) -> &T {
        &self.data[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }

    pub fn x(&self) -> &T {
        self.get(0)
    }

    pub fn x_mut(&mut self) -> &mut T {
        self.get_mut(0)
    }

    pub fn y(&self) -> &T {
        self.get(1)
    }

    pub fn y_mut(&mut self) -> &mut T {
        self.get_mut(1)
    }

    pub fn z(&self) -> &T {
        self.get(2)
    }

    pub fn z_mut(&mut self) -> &mut T {
        self.get_mut(2)
    }

    pub fn w(&self) -> &T {
        self.get(3)
    }

    pub fn w_mut(&mut self) -> &mut T {
        self.get_mut(3)
    }

    // TODO to_matrix

    pub fn length_squared(&self) -> T {
        let mut n = T::additive_identity();

        for i in 0..self.data.len() {
            let v = self.data[i];
            n = v.mul_add(&v, &n);
        }
        n
    }

    pub fn length(&self) -> T {
        self.length_squared().sqrt()
    }

    pub fn normalize(&mut self, length: T) -> &mut Self {
        let len = self.length();

        for i in 0..self.data.len() {
            self.data[i] = (self.data[i] * length) / len;
        }
        self
    }

    pub fn dot(&self, other: &Vector<T>) -> T {
        let mut n = T::additive_identity();

        // TODO Check other's size
        for i in 0..self.data.len() {
            n = self.data[i].mul_add(&other.data[i], &n);
        }
        n
    }

    pub fn cross_product(&self, other: &Vector<T>) -> Self {
        // TODO Assert that size is `3`
        Self::from_vec(vec!{
            self.get(1).mul_add(other.get(2), &-(*self.get(2) * *other.get(1))),
            self.get(2).mul_add(other.get(0), &-(*self.get(0) * *other.get(2))),
            self.get(0).mul_add(other.get(1), &-(*self.get(1) * *other.get(0))),
        })
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

impl<T: Field<T>> std::ops::Neg for Vector::<T> {
    type Output = Vector::<T>;

    fn neg(self) -> Self::Output {
        let mut v = self.clone();
        v.negate();
        v
    }
}

impl<T: Field<T>> std::ops::Add<T> for Vector::<T> {
    type Output = Vector::<T>;

    fn add(self, n: T) -> Self::Output {
        let mut v = self.clone();
        v.add_val(&n);
        v
    }
}

impl<T: Field<T>> std::ops::Add<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn add(self, n: Vector::<T>) -> Self::Output {
        let mut v = self.clone();
        v.add_self(&n);
        v
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

    fn sub(self, n: T) -> Self::Output {
        let mut v = self.clone();
        v.subtract_val(&n);
        v
    }
}

impl<T: Field<T>> std::ops::Sub<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn sub(self, n: Vector::<T>) -> Self::Output {
        let mut v = self.clone();
        v.subtract_self(&n);
        v
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

    fn mul(self, n: T) -> Self::Output {
        let mut v = self.clone();
        v.multiply_val(&n);
        v
    }
}

impl<T: Field<T>> std::ops::Mul<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn mul(self, n: Vector::<T>) -> Self::Output {
        let mut v = self.clone();
        v.multiply_self(&n);
        v
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

    fn div(self, n: T) -> Self::Output {
        let mut v = self.clone();
        v.divide_val(&n);
        v
    }
}

impl<T: Field<T>> std::ops::Div<Vector::<T>> for Vector::<T> {
    type Output = Vector::<T>;

    fn div(self, n: Vector::<T>) -> Self::Output {
        let mut v = self.clone();
        v.divide_self(&n);
        v
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
            write!(f, "{}", *self.get(i));
            if i < self.get_size() - 1 {
                write!(f, " ");
            }
        }
        write!(f, "")
    }
}

macro_rules! assert_eq_delta {
	($n0:expr, $n1:expr) => {
		let r = ($n0).epsilon_equal(&($n1));
		if !r {
			eprintln!("Delta assert fail: {} != {}", $n0, $n1);
		}
		assert!(r);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

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
		assert!(mat.get_height() == 4);
		assert!(mat.get_width() == 3);
		assert_eq_delta!(*mat.get(0, 0), 1.);
		assert_eq_delta!(*mat.get(1, 0), 2.);
		assert_eq_delta!(*mat.get(1, 2), 3.);
		assert_eq_delta!(*mat.get(0, 1), 0.);
		assert_eq_delta!(*mat.get(2, 1), 0.);

		mat.transpose();

		assert!(mat.is_transposed());
		assert!(mat.get_height() == 3);
		assert!(mat.get_width() == 4);
		assert_eq_delta!(*mat.get(0, 0), 1.);
		assert_eq_delta!(*mat.get(0, 1), 2.);
		assert_eq_delta!(*mat.get(2, 1), 3.);
		assert_eq_delta!(*mat.get(1, 0), 0.);
		assert_eq_delta!(*mat.get(1, 2), 0.);
	}

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

	// TODO Rank

	#[test]
	fn test_mat_trace0() {
		let mut mat = Matrix::<f64>::new(4, 3);
		mat += 1.;
		assert_eq_delta!(mat.trace(), 3. as f64);
	}

	#[test]
	fn test_mat_solve0() {
		let mut mat = Matrix::<f64>::from_vec(3, 4, vec!{
			1., 0., 0., 1.,
			0., 1., 0., 1.,
			0., 0., 1., 1.,
		});
		let r = mat.solve();
		for i in 0..r.get_size() {
			assert_eq_delta!(*r.get(i), 1.);
		}
	}

	#[test]
	fn test_mat_solve1() {
		let mut mat = Matrix::<f64>::from_vec(2, 3, vec!{
			4., 2., -1.,
			3., -1., 2.,
		});
		let r = mat.solve();
		assert_eq!(r.get_size(), 2);
		assert!((*r.get(0) - 0.3).abs() < 0.000001);
		assert!((*r.get(1) - -1.1).abs() < 0.000001);
	}

	#[test]
	fn test_mat_solve2() {
		let mut mat = Matrix::<f64>::from_vec(3, 4, vec!{
			3., 2., -1., 1.,
			2., -2., 4., -2.,
			-1., 0.5, -1., 0.,
		});
		let r = mat.solve();
		assert_eq!(r.get_size(), 3);
		assert!((*r.get(0) - 1.).abs() < 0.000001);
		assert!((*r.get(1) - -2.).abs() < 0.000001);
		assert!((*r.get(2) - -2.).abs() < 0.000001);
	}

	#[test]
	fn test_mat_solve3() {
		let mut mat = Matrix::<f64>::from_vec(2, 4, vec!{
			1., 1., 1., 1.,
			1., 1., 2., 3.,
		});
		let r = mat.solve();
		assert_eq!(r.get_size(), 3);

		let a = mat.submatrix(0, 0, 2, 3) * r;
		assert_eq!(a.get_size(), 2);
		assert!((*a.get(0) - 1.).abs() < 0.000001);
		assert!((*a.get(1) - 3.).abs() < 0.000001);
	}

	/*#[test]
	fn test_mat_solve4() {
		let mut mat = Matrix::<f64>::new(100, 101);
        for i in 0..mat.get_height() {
            for j in 0..mat.get_width() {
                *mat.get_mut(i, j) = 1.;
            }
        }

		let r = mat.solve();
		assert_eq_delta!(r.get_size(), 100);

		let a = mat.submatrix(0, 0, 100, 100) * r;
		assert_eq_delta!(a.get_size(), 100);
        for i in 0..a.get_size() {
            assert!((*a.get(i) - 1.).abs() < 0.000001);
        }
	}*/

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
