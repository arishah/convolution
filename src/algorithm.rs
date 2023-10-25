use stack_alg_sim::olken::LRUSplay;
use stack_alg_sim::LRU;
use std::cmp::min;
use std::env;
use std::f64::consts::PI;
use std::time::Instant;

static mut GLOBAL_DMC: f64 = 0.0;

pub fn convolution(n: usize, image_i: Vec<f32>, k: usize, image_k: Vec<f32>, result: &mut Vec<f32>, c: usize, batch_sz: usize) {
    let now = Instant::now();

    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;
    let mut analyzer: LRUSplay<(char, usize, usize)> = LRUSplay::<(char, usize, usize)>::new();

    for p in 0..num_passes {
        for i in 0..(n - k + 1) {
            for j in 0..(n - k + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..k {
                        for x in 0..k {
                            sum += image_k[y * k + x] * image_i[((i + y) * n + (j + x)) * c + l];
                            let cur_k = analyzer.rec_access(('k', y, x));
                            unsafe { GLOBAL_DMC += (cur_k.unwrap_or(0) as f64).sqrt();}
                            let cur_i = analyzer.rec_access(('i', (i + y)* c + l, (j + x) * c + l));
                            unsafe { GLOBAL_DMC += (cur_i.unwrap_or(0) as f64).sqrt();}
                        }
                    }
                }
                result[i * n + j] += sum;
                let cur_i = analyzer.rec_access(('r', i, j));
                unsafe { GLOBAL_DMC += (cur_i.unwrap_or(0) as f64).sqrt();}
            }
        }
    }

    let elapsed = now.elapsed();
    println!("n, {}, k, {}, c, {}, b, {}, t, {:.2?}", n, k, c, batch_sz, elapsed);
}