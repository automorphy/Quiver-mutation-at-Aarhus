use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Quiver<const N: usize> {
    pub b: [[i128; N]; N],
}

impl<const N: usize> Quiver<N> {
    pub fn new(matrix: [[i128; N]; N]) -> Self {
        Quiver { b: matrix }
    }

    pub fn from_upper_tri(upper_tri: &[i128]) -> Self {
        let mut b = [[0; N]; N];
        let mut k = 0;
        for i in 0..N {
            for j in i + 1..N {
                b[i][j] = upper_tri[k];
                b[j][i] = -upper_tri[k];
                k += 1;
            }
        }
        Quiver { b }
    }

    pub fn to_upper_tri(&self) -> Vec<i128> {
        let mut vec = Vec::new();
        for i in 0..N {
            for j in i + 1..N {
                vec.push(self.b[i][j]);
            }
        }
        vec
    }

    pub fn mutate(&self, k: usize) -> Self {
        let mut next_b = [[0i128; N]; N];
        let b = self.b;

        for i in 0..N {
            for j in 0..N {
                if i == k || j == k {
                    next_b[i][j] = -b[i][j];
                } else {
                    let mut delta = b[i][k] * b[k][j];
                    if delta < 0 {
                        delta = 0;
                    }
                    next_b[i][j] = b[i][j] + (b[i][k].signum() * delta);
                }
            }
        }
        Quiver { b: next_b }
    }

    pub fn get_canonical_vector(&self) -> Vec<i128> {
        let mut best = self.to_upper_tri();
        let mut indices: Vec<usize> = (0..N).collect();
        
        loop {
            let mut candidate_vec = Vec::new();
            for i in 0..N {
                for j in i + 1..N {
                    let u = indices[i];
                    let v = indices[j];
                    candidate_vec.push(self.b[u][v]);
                }
            }
            if candidate_vec < best {
                best = candidate_vec;
            }
            if !next_permutation(&mut indices) { break; }
        }
        best
    }

    pub fn get_digraph_class_and_aligned_weights(&self) -> (Vec<i8>, Vec<i128>) {
        // 1. Extract Skeleton (Sign Matrix)
        let mut sign_matrix = [[0i8; N]; N];
        for i in 0..N {
            for j in 0..N {
                sign_matrix[i][j] = self.b[i][j].signum() as i8;
            }
        }

        // 2. Find Canonical Permutation for Skeleton
        let mut best_sign_vec = Vec::new();
        // Initialize with identity permutation
        for i in 0..N {
            for j in i + 1..N {
                best_sign_vec.push(sign_matrix[i][j]);
            }
        }
        
        let mut best_perm = (0..N).collect::<Vec<_>>();
        let mut indices: Vec<usize> = (0..N).collect();

        loop {
            let mut candidate_sign_vec = Vec::new();
            for i in 0..N {
                for j in i + 1..N {
                    let u = indices[i];
                    let v = indices[j];
                    candidate_sign_vec.push(sign_matrix[u][v]);
                }
            }
            
            // Lexicographical comparison for canonical skeleton
            if candidate_sign_vec < best_sign_vec {
                best_sign_vec = candidate_sign_vec;
                best_perm = indices.clone();
            }
            
            if !next_permutation(&mut indices) { break; }
        }

        // 3. Apply Best Permutation to Original Weights
        let mut aligned_weights = Vec::new();
        for i in 0..N {
            for j in i + 1..N {
                let u = best_perm[i];
                let v = best_perm[j];
                aligned_weights.push(self.b[u][v]);
            }
        }

        (best_sign_vec, aligned_weights)
    }

    pub fn max_weight(&self) -> i128 {
        let mut m = 0;
        for row in &self.b {
            for val in row {
                m = m.max(val.abs());
            }
        }
        m
    }
}

// Rank 3 Specific logic
impl Quiver<3> {
    pub fn markov_constant(&self) -> i128 {
        // C = x^2 + y^2 + z^2 - xyz
        // We use absolute values to represent the 'counts of arrows' in the cycle
        // regardless of the specific orientation in the b-matrix.
        let x = self.b[0][1].abs();
        let y = self.b[1][2].abs();
        let z = self.b[0][2].abs(); 
        
        x*x + y*y + z*z - x*y*z
    }

    pub fn is_acyclic_paper_criteria(&self) -> bool {
        let c = self.markov_constant();
        let x = self.b[0][1].abs();
        let y = self.b[1][2].abs();
        let z = self.b[0][2].abs();
        let min_v = x.min(y).min(z);
        
        // Theorem 1.1: Cluster-acyclic if C > 4 or min < 2
        c > 4 || min_v < 2
    }
}

// Rank 4 Specific logic
impl Quiver<4> {
    pub fn nma_reason(&self) -> Option<String> {
        let indices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        for sub in indices {
            let x = self.b[sub[0]][sub[1]].abs();
            let y = self.b[sub[1]][sub[2]].abs();
            let z = self.b[sub[0]][sub[2]].abs();
            
            if x >= 2 && y >= 2 && z >= 2 {
                let c = x*x + y*y + z*z - x*y*z;
                if c <= 4 {
                    return Some(format!("Markov cyclic subquiver indices {:?}", sub));
                }
            }
        }
        None
    }
}

fn next_permutation(indices: &mut [usize]) -> bool {
    let len = indices.len();
    if len <= 1 { return false; }
    let mut i = len - 1;
    while i > 0 && indices[i - 1] >= indices[i] {
        i -= 1;
    }
    if i == 0 { return false; }
    let mut j = len - 1;
    while indices[j] <= indices[i - 1] {
        j -= 1;
    }
    indices.swap(i - 1, j);
    indices[i..].reverse();
    true
}