use crate::Field;

#[derive(Clone)]
pub struct Complex<T>
{
	pub x: T,
	pub y: T,
}

impl<T: Field<T>> Complex::<T> {
	pub fn new(x: &T, y: &T) -> Self {
		Self {
			x: *x,
			y: *y,
		}
	}

	pub fn conjugate(&self) -> Self {
		let mut c = self.clone();
		c.y = -c.y;
		c
	}

	pub fn modulus(&self) -> T {
		(self.x * self.x + self.y * self.y).sqrt()
	}

	pub fn argument(&self) -> T {
		self.y.atan2(&self.x)
	}
}

impl<T: Field<T>> std::ops::Neg for Complex::<T> {
	type Output = Complex::<T>;

	fn neg(self) -> Self::Output {
		let mut c = self.clone();
		c.x = -c.x;
		c.y = -c.y;
		c
	}
}

impl<T: Field<T>> std::ops::Add<Complex::<T>> for Complex::<T> {
	type Output = Complex::<T>;

	fn add(self, n: Complex::<T>) -> Self::Output {
		let mut c = self.clone();
		c.x += n.x;
		c.y += n.y;
		c
	}
}

impl<T: Field<T>> std::ops::Add<T> for Complex::<T> {
	type Output = Complex::<T>;

	fn add(self, n: T) -> Self::Output {
		let mut c = self.clone();
		c.x += n;
		c
	}
}

impl<T: Field<T>> std::ops::AddAssign<Complex::<T>> for Complex::<T> {
	fn add_assign(&mut self, n: Complex::<T>) {
		self.x += n.x;
		self.y += n.y;
	}
}

impl<T: Field<T>> std::ops::AddAssign<T> for Complex::<T> {
	fn add_assign(&mut self, n: T) {
		self.x += n;
	}
}

impl<T: Field<T>> std::ops::Sub<Complex::<T>> for Complex::<T> {
	type Output = Complex::<T>;

	fn sub(self, n: Complex::<T>) -> Self::Output {
		let mut c = self.clone();
		c.x -= n.x;
		c.y -= n.y;
		c
	}
}

impl<T: Field<T>> std::ops::Sub<T> for Complex::<T> {
	type Output = Complex::<T>;

	fn sub(self, n: T) -> Self::Output {
		let mut c = self.clone();
		c.x -= n;
		c
	}
}

impl<T: Field<T>> std::ops::SubAssign<Complex::<T>> for Complex::<T> {
	fn sub_assign(&mut self, n: Complex::<T>) {
		self.x -= n.x;
		self.y -= n.y;
	}
}

impl<T: Field<T>> std::ops::SubAssign<T> for Complex::<T> {
	fn sub_assign(&mut self, n: T) {
		self.x -= n;
	}
}

impl<T: Field<T>> std::ops::Mul<Complex::<T>> for Complex::<T> {
	type Output = Complex::<T>;

	fn mul(self, n: Complex::<T>) -> Self::Output {
		let mut c = self.clone();
		let x = c.x;
		c.x = x * n.x - c.y * n.y;
		c.y = x * n.y + c.y * n.x;
		c
	}
}

impl<T: Field<T>> std::ops::Mul<T> for Complex::<T> {
	type Output = Complex::<T>;

	fn mul(self, n: T) -> Self::Output {
		let mut c = self.clone();
		c.x *= n;
		c.y *= n;
		c
	}
}

impl<T: Field<T>> std::ops::MulAssign<Complex::<T>> for Complex::<T> {
	fn mul_assign(&mut self, n: Complex::<T>) {
		let x = self.x;
		self.x = x * n.x - self.y * n.y;
		self.y = x * n.y + self.y * n.x;
	}
}

impl<T: Field<T>> std::ops::MulAssign<T> for Complex::<T> {
	fn mul_assign(&mut self, n: T) {
		self.x *= n;
		self.y *= n;
	}
}

impl<T: Field<T>> std::ops::Div<Complex::<T>> for Complex::<T> {
	type Output = Complex::<T>;

	fn div(self, n: Complex::<T>) -> Self::Output {
		let mut c = self.clone();
		let x = c.x;
		let div = n.x * n.x + n.y * n.y;
		c.x = (x * n.x + c.y * n.y) / div;
		c.y = (c.y * n.x - x * n.y) / div;
		c
	}
}

impl<T: Field<T>> std::ops::Div<T> for Complex::<T> {
	type Output = Complex::<T>;

	fn div(self, n: T) -> Self::Output {
		let mut c = self.clone();
		c.x /= n;
		c.y /= n;
		c
	}
}

impl<T: Field<T>> std::ops::DivAssign<Complex::<T>> for Complex::<T> {
	fn div_assign(&mut self, n: Complex::<T>) {
		let div = n.x * n.x + n.y * n.y;
		let x = self.x;
		self.x = (x * n.x + self.y * n.y) / div;
		self.y = (self.y * n.x - x * n.y) / div;
	}
}

impl<T: Field<T>> std::ops::DivAssign<T> for Complex::<T> {
	fn div_assign(&mut self, n: T) {
		self.x /= n;
		self.y /= n;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_complex_mul0() {
		let c = Complex::<f64>::new(&1., &0.) * 2.;
		assert_eq!(c.x, 2 as f64);
		assert_eq!(c.y, 0 as f64);
	}

	#[test]
	fn test_complex_mul1() {
		let c = Complex::<f64>::new(&1., &1.) * 2.;
		assert_eq!(c.x, 2 as f64);
		assert_eq!(c.y, 2 as f64);
	}

	#[test]
	fn test_complex_mul2() {
		let c0 = Complex::<f64>::new(&1., &1.);
		let c1 = Complex::<f64>::new(&2., &2.);
		let c2 = c0 * c1;
		assert_eq!(c2.x, 0 as f64);
		assert_eq!(c2.y, 4 as f64);
	}

	#[test]
	fn test_complex_div0() {
		let c = Complex::<f64>::new(&1., &0.) / 2.;
		assert_eq!(c.x, 0.5 as f64);
		assert_eq!(c.y, 0 as f64);
	}

	#[test]
	fn test_complex_div1() {
		let c = Complex::<f64>::new(&1., &1.) / 2.;
		assert_eq!(c.x, 0.5 as f64);
		assert_eq!(c.y, 0.5 as f64);
	}

	#[test]
	fn test_complex_div2() {
		let c0 = Complex::<f64>::new(&1., &1.);
		let c1 = Complex::<f64>::new(&2., &2.);
		let c2 = c0 / c1;
		assert_eq!(c2.x, 0.5 as f64);
		assert_eq!(c2.y, 0. as f64);
	}
}
