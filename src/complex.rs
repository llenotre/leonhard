use crate::Value;

#[derive(Clone)]
struct Complex<T>
{
	x: T,
	y: T,
}

impl<T: Value<T>> Complex::<T> {
	pub fn conjugate(&mut self) -> &mut Self {
		self.y = -self.y;
		self
	}
}

impl<T: Value<T>> std::ops::Neg for Complex::<T> {
	type Output = Complex::<T>;

	fn neg(self) -> Self::Output {
		let mut c = self.clone();
		c.x = -c.x;
		c.y = -c.y;
		c
	}
}

impl<T: Value<T>> std::ops::Add<Complex::<T>> for Complex::<T> {
	type Output = Complex::<T>;

	fn add(self, n: Complex::<T>) -> Self::Output {
		let mut c = self.clone();
		c.x += n.x;
		c.y += n.y;
		c
	}
}

impl<T: Value<T>> std::ops::AddAssign<Complex::<T>> for Complex::<T> {
	fn add_assign(&mut self, n: Complex::<T>) {
		self.x += n.x;
		self.y += n.y;
	}
}

impl<T: Value<T>> std::ops::Sub<Complex::<T>> for Complex::<T> {
	type Output = Complex::<T>;

	fn sub(self, n: Complex::<T>) -> Self::Output {
		let mut c = self.clone();
		c.x -= n.x;
		c.y -= n.y;
		c
	}
}

impl<T: Value<T>> std::ops::SubAssign<Complex::<T>> for Complex::<T> {
	fn sub_assign(&mut self, n: Complex::<T>) {
		self.x -= n.x;
		self.y -= n.y;
	}
}

impl<T: Value<T>> std::ops::Mul<Complex::<T>> for Complex::<T> {
	type Output = Complex::<T>;

	fn mul(self, n: Complex::<T>) -> Self::Output {
		let mut c = self.clone();
		let x = c.x;
		c.x = x * n.x - c.y * n.y;
		c.y = x * n.y + c.y * n.x;
		c
	}
}

impl<T: Value<T>> std::ops::MulAssign<Complex::<T>> for Complex::<T> {
	fn mul_assign(&mut self, n: Complex::<T>) {
		let x = self.x;
		self.x = x * n.x - self.y * n.y;
		self.y = x * n.y + self.y * n.x;
	}
}

impl<T: Value<T>> std::ops::Div<Complex::<T>> for Complex::<T> {
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

impl<T: Value<T>> std::ops::DivAssign<Complex::<T>> for Complex::<T> {
	fn div_assign(&mut self, n: Complex::<T>) {
		let div = n.x * n.x + n.y * n.y;
		let x = self.x;
		self.x = (x * n.x + self.y * n.y) / div;
		self.y = (self.y * n.x - x * n.y) / div;
	}
}
