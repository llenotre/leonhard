#![allow(dead_code)]

mod complex;
mod linear_algebra;
mod polynomial;

trait Value<T>: Default + Copy
	+ std::ops::Neg<Output = T>
	+ std::ops::Add<Output = T>
	+ std::ops::AddAssign
	+ std::ops::Sub<Output = T>
	+ std::ops::SubAssign
	+ std::ops::Mul<Output = T>
	+ std::ops::MulAssign
	+ std::ops::Div<Output = T>
	+ std::ops::DivAssign {}
