use stack_alg_sim::olken::LRUSplay;
use stack_alg_sim::LRU;
use std::cmp::min;
use std::env;
use std::f64::consts::PI;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

pub fn convolution(n: usize, image_i: Vec<f32>, k: usize, image_k: Vec<f32>, result: &mut Vec<f32>, c: usize, batch_sz: usize) {
    let now = Instant::now();

    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;

    let res = Arc::new(Mutex::new(vec![0.0; n*n]));
    let c_res = Arc::clone(&res);

    for p in 0..num_passes {
        (0..(n - k + 1)).into_par_iter().for_each(|i| {
            for j in 0..(n - k + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..k {
                        for x in 0..k {
                            sum += image_k[y * k + x] * image_i[((i + y) * n + (j + x)) + l*n^2];
                        }
                    }
                }
                c_res.lock().unwrap()[i * n + j] += sum;
            }
        });
    }

    let mut r = res.lock().unwrap().clone();
    result.copy_from_slice(&r);

    let elapsed = now.elapsed();
    println!("n, {}, k, {}, c, {}, b, {}, t, {:.2?}", n, k, c, batch_sz, elapsed);
}