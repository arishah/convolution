#![feature(linked_list_remove)]
use std::cmp::min;
use std::collections::LinkedList;
use std::env;
use num_complex::Complex;
use std::f64::consts::PI;
use rand::Rng;

const N: usize = 128;
const K: usize = 3;

static mut GLOBAL_DMC: f64 = 0.0;

unsafe fn update_dmc(sampler: &mut LinkedList<u64>, addr_ptr: *const f32) {
    let addr = addr_ptr as u64;

    let mut pos: usize = 0;
    let mut it_prev = sampler.iter_mut().peekable();
    while let Some(val) = it_prev.next() {
        if *val == addr {
            it_prev.next();
            GLOBAL_DMC += (pos as f64).sqrt();
            break;
        }
        pos += 1;
    }
    if pos != sampler.len() {
        sampler.remove(pos as usize);
    }
    sampler.push_front(addr);
}

unsafe fn update_dmc_complex(sampler: &mut LinkedList<u64>, addr_ptr: *const Complex<f64>) {
    let addr = addr_ptr as u64;

    let mut pos: usize = 0;
    let mut it_prev = sampler.iter_mut().peekable();
    while let Some(val) = it_prev.next() {
        if *val == addr {
            it_prev.next();
            GLOBAL_DMC += (pos as f64).sqrt();
            break;
        }
        pos += 1;
    }
    if pos != sampler.len() {
        sampler.remove(pos as usize);
    }
    sampler.push_front(addr);
}

fn conv2d(image_i: &mut [f32], image_k: &mut [f32], result: &mut [f32], c: usize, batch_sz: usize) {
    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;
    let mut sampler: LinkedList<u64> = LinkedList::new();

    for p in 0..num_passes {
        for i in 0..(N - K + 1) {
            for j in 0..(N - K + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..K {
                        for x in 0..K {
                            sum += image_k[y * K + x] * image_i[((i + y) * N + (j + x)) * c + l];
                            unsafe {
                                update_dmc(&mut sampler, &image_k[y * K + x]);
                                update_dmc(&mut sampler, &image_i[((i + y) * N + (j + x)) * c + l]);
                            }
                        }
                    }
                }
                result[i * N + j] += sum;
                unsafe {
                    update_dmc(&mut sampler, &result[i * N + j]);
                }
            }
        }
    }
}

fn fft(f: &mut Vec<Complex<f64>>, ln: usize, invert: bool) {
    let n = 1 << ln;
    let half_n = n / 2;
    let mut j = 1;
    let mut sampler: LinkedList<u64> = LinkedList::new();

    for i in 1..n {
        if i < j {
            f.swap(i - 1, j - 1);
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
                unsafe {
                    update_dmc_complex(&mut sampler, &f[m - 1]);
                    update_dmc_complex(&mut sampler, &f[i - 1]);
                }
            }
            u *= w;
        }
    }

    if invert {
        for elem in f.iter_mut() {
            *elem /= n as f64;
        }
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
    conv2d(&mut image_i, &mut image_k, &mut result, channels, batch_sz as usize);
    println!("DMC: {}", unsafe { GLOBAL_DMC });
    } else {
    let vec_size = args[2].parse::<usize>().unwrap();
    let len = args[3].parse::<usize>().unwrap();
    let mut rng = rand::thread_rng();
    let mut f: Vec<Complex<f64>> = (0..vec_size)
    .map(|_| {
        let real_part = rng.gen_range(-1.0..1.0); // Adjust the range as needed
        let imag_part = rng.gen_range(-1.0..1.0); // Adjust the range as needed
        Complex::new(real_part, imag_part)
    })
    .collect();

    for complex in &f {
        println!("before: {}", complex);
    }

    fft(&mut f, len, false);

    println!("DMC: {}", unsafe { GLOBAL_DMC });

/*     for complex in &f {
        println!("after: {}", complex);
    }*/
}

    return ();
}
