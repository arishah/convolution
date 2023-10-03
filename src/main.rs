#![feature(linked_list_remove)]
use std::cmp::min;
use std::env;
use num_complex::Complex;
use std::f64::consts::PI;
use rand::Rng;
use stack_alg_sim::olken::LRUSplay;
use stack_alg_sim::LRU;

const N: usize = 128;
const K: usize = 3;

static mut GLOBAL_DMC: f64 = 0.0;

fn conv(image_i: &mut [f32], image_k: &mut [f32], result: &mut [f32], c: usize, batch_sz: usize) {
    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;
    let mut analyzer: LRUSplay::<(char, usize)> = LRUSplay::<(char, usize)>::new();

    for p in 0..num_passes {
        for i in 0..(N - K + 1) {
            for j in 0..(N - K + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..K {
                        for x in 0..K {
                            sum += image_k[y * K + x] * image_i[((i + y) * N + (j + x)) * c + l];
                            let cur_k = analyzer.rec_access(('k', y * K + x));
                            unsafe { GLOBAL_DMC += (cur_k.unwrap_or(0) as f64).sqrt();}
                            let cur_i = analyzer.rec_access(('i', ((i + y) * N + (j + x)) * c + l));
                            unsafe { GLOBAL_DMC += (cur_i.unwrap_or(0) as f64).sqrt();}
                        }
                    }
                }
                result[i * N + j] += sum;
                let cur_i = analyzer.rec_access(('r', i * N + j));
                unsafe { GLOBAL_DMC += (cur_i.unwrap_or(0) as f64).sqrt();}
            }
        }
    }
}

fn fft_it(f: &mut Vec<Complex<f64>>, ln: usize, invert: bool) -> f64 {
    let n = 1 << ln;
    let half_n = n / 2;
    let mut j = 1;
    let mut analyzer = LRUSplay::<usize>::new();
    let mut dmc: f64 = 0.0;

    for i in 1..n {
        if i < j {
            f.swap(i - 1, j - 1);
            let mut cur = analyzer.rec_access(i-1);
            dmc += (cur.unwrap_or(0) as f64).sqrt();
            cur = analyzer.rec_access(j-1);
            dmc += (cur.unwrap_or(0) as f64).sqrt();
        }
        let mut k = half_n;
        while k < j {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    for l in 1..=ln {
        let k = 1 << l;
        let half_k = k / 2;
        let mut u = Complex::new(1.0, 0.0);
        let mut w = Complex::from_polar(1.0, -PI / half_k as f64);

        if invert {
            w = w.conj();
        }

        for j in 1..=half_k {
            for i in (j..=n).step_by(k) {
                let m = i + half_k;
                let t = f[m - 1] * u;
                f[m - 1] = f[i - 1] - t;
                f[i - 1] = f[i - 1] + t;
                let mut cur = analyzer.rec_access(m-1);
                dmc += (cur.unwrap_or(0) as f64).sqrt();
                cur = analyzer.rec_access(i-1);
                dmc += (cur.unwrap_or(0) as f64).sqrt();
            }
            u *= w;
        }
    }

    if invert {
        for elem in f.iter_mut() {
            *elem /= n as f64;
        }
    }
    dmc
}

fn fft_recursive(n: usize, mut a: Vec<Complex<f64>>, analyzer: &mut LRUSplay::<(char, usize)>) -> Vec<Complex<f64>> {
    if n == 1 {
        a
    } else {
        let n_half = n / 2;

        let evens: Vec<Complex<f64>> = a.iter().step_by(2).cloned().collect();

        for i in 0..n_half {
            let cur_e = analyzer.rec_access(('a', i * 2));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
            let cur_e = analyzer.rec_access(('1', i));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
        }

        let mut f_even = fft_recursive(evens.len(), evens, analyzer);

        for i in 0..n_half {
            let cur_e = analyzer.rec_access(('2', i));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
            let cur_o = analyzer.rec_access(('e', i));
            unsafe { GLOBAL_DMC += (cur_o.unwrap_or(0) as f64).sqrt();}
        }
        let odds: Vec<Complex<f64>> = a.iter().skip(1).step_by(2).cloned().collect();

        for i in 0..n_half {
            let cur_e = analyzer.rec_access(('a', i * 2 + 1));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
            let cur_e = analyzer.rec_access(('3', i));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
        }
        let mut f_odd = fft_recursive(odds.len(), odds, analyzer);

        for i in 0..n_half {
            let cur_e = analyzer.rec_access(('4', i));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
            let cur_o = analyzer.rec_access(('o', i));
            unsafe { GLOBAL_DMC += (cur_o.unwrap_or(0) as f64).sqrt();}
        }

        f_even.append(&mut f_odd);
        a = f_even;

        for i in 0..n_half {
            let cur_o = analyzer.rec_access(('e', i));
            unsafe { GLOBAL_DMC += (cur_o.unwrap_or(0) as f64).sqrt();}
            let cur_e = analyzer.rec_access(('a', i));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
        }
        for i in 0..n_half {
            let cur_o = analyzer.rec_access(('o', i));
            unsafe { GLOBAL_DMC += (cur_o.unwrap_or(0) as f64).sqrt();}
            let cur_e = analyzer.rec_access(('a', n_half + i + 1));
            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
        }

        for k in 0..(n_half) {
            let w: Complex<f64> = Complex::<f64>{re: 0., im:((-(2.) * std::f64::consts::PI * k as f64) / n as f64)}.exp();
            let t = a[k];

            let mut cur_a = analyzer.rec_access(('a', k));
            unsafe { GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();}

            a[k] = t + w * a[k+(n /2)];

            cur_a = analyzer.rec_access(('a', k+(n /2)));
            unsafe { GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();}
            cur_a = analyzer.rec_access(('a', k));
            unsafe { GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();}

            a[k + n/2] = t - w * a[k + n/2];

            cur_a = analyzer.rec_access(('a', k+(n /2)));
            unsafe { GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();}
            cur_a = analyzer.rec_access(('a', k+(n / 2)));
            unsafe { GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();}
        }
        a
    }
}



fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args[1].parse::<String>().unwrap();

    if mode == "conv" {
    let channels = args[2].parse::<usize>().unwrap();
    let batch_sz: i32 = args[3].parse::<i32>().unwrap();
    let batch_sz = if batch_sz == -1 {
        channels as i32
    } else {
        batch_sz as i32
    };

    let mut image_i = vec![0.0; N * N * channels];
    let mut result = vec![0.0; N * N];
    let mut image_k = vec![0.0; K * K];

    conv(&mut image_i, &mut image_k, &mut result, channels, batch_sz as usize);
    println!("DMC: {}", unsafe{ GLOBAL_DMC});
    } else {
    let vec_size = args[2].parse::<usize>().unwrap();
    let len = args[3].parse::<usize>().unwrap();
    let mut rng = rand::thread_rng();
    let mut a: Vec<Complex<f64>> = (0..vec_size)
    .map(|_| {
        let real_part = rng.gen_range(-1.0..1.0);
        let imag_part = rng.gen_range(-1.0..1.0);
        Complex::new(real_part, imag_part)
    })
    .collect();

//    let dmc = fft_it(&mut a, len, false);
//    println!("DMC: {}", dmc);

    let mut analyzer: LRUSplay::<(char, usize)> = LRUSplay::<(char, usize)>::new();
    fft_recursive(a.len(), a, &mut analyzer);
    println!("DMC: {}", unsafe { GLOBAL_DMC });
}
    return ();
}
