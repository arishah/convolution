use rayon::prelude::*;
use stack_alg_sim::olken::LRUSplay;
use stack_alg_sim::LRU;
use std::cmp::min;
use std::env;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub fn convolution(
    n: usize,
    image_i: Vec<Vec<f64>>,
    k: usize,
    image_k: Vec<Vec<f64>>,
//    result: Vec<Vec<f64>>,
    c: usize,
    batch_sz: usize,
) -> Vec<Vec<f64>> {
    let now = Instant::now();

    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;

    let res = Arc::new(Mutex::new(vec![vec![0.0; n]; n]));
    let c_res = Arc::clone(&res);

    rayon::ThreadPoolBuilder::new().num_threads(10).build_global().unwrap();

    for p in 0..num_passes {
        (0..(n - k + 1)).into_par_iter().for_each(|i| {
            for j in 0..(n - k + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..k {
                        for x in 0..k {
                            sum += image_k[y][x] * image_i[y+i + l*n][j + x + l*n];
                        }
                    }
                }
                c_res.lock().unwrap()[i][j] += sum;
            }
        });
    }

//    let mut r = res.lock().unwrap().clone();
//    result.copy_from_slice(&r);
    let mut result = res.lock().unwrap().clone();

    let elapsed = now.elapsed();
    println!(
        "n, {}, k, {}, c, {}, b, {}, t, {:.2?}",
        n, k, c, batch_sz, elapsed
    );
    return result;
}

pub fn matrix_multi(n: Vec<Vec<i32>>, m: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let n_1 = n.len();
    let n_2 = n[0].len();
    let m_2 = m[0].len();

    let mut result = vec![vec![0; m_2]; n_1];

    for i in 0..n_1 {
        for j in 0..m_2 {
            for k in 0..n_2 {
                result[i][j] += n[i][k] * m[k][j];
            }
        }
    }

    result
}