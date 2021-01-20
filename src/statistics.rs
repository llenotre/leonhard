use crate::linear_algebra::*;

// TODO Generic
pub fn covariance_matrix(values: &Vec::<Vector::<f64>>) -> Matrix::<f64> {
	// TODO Check that at least there is a least one entry
	// TODO Check that every vectors have the same dimension

	let dim = values[0].get_size();
	let mut mat = Matrix::<f64>::new(dim, dim);

	for i in 0..dim {
		for j in 0..dim {
			for v in values {
				*mat.get_mut(i, j) += *v.get(i) * *v.get(j);
			}
			*mat.get_mut(i, j) /= values.len() as f64;
		}
	}

	mat
}

// TODO tests
