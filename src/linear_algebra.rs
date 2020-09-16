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

    pub fn get_data(&self) -> &Vec<T> {
        &self.data
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
            n = v.mul_add(&v, &n);
        }
        n
    }

    pub fn length(&self) -> T {
        self.length_squared().sqrt()
    }

    pub fn normalize(&mut self, length: T) -> &mut Self {
        let len = self.length();

        for i in 0..self.size {
            self.data[i] = (self.data[i] * length) / len;
        }
        self
    }

    pub fn dot(&self, other: &Vector<T>) -> T {
        let mut n = T::additive_identity();

        // TODO Check other's size
        for i in 0..self.size {
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
