use num_complex::Complex;
use rand::Rng;
use stack_alg_sim::olken::LRUSplay;
use stack_alg_sim::LRU;
use std::cmp::min;
use std::env;
use std::f64::consts::PI;
use hist::Hist;
use std::mem;

//const N: usize = 128;
//const K: usize = 3;

static mut GLOBAL_DMC: f64 = 0.0;
static mut ITER: i32 = 0;



pub fn conv(
    n: usize,
    image_i: Vec<f32>,
    k: usize,
    image_k: Vec<f32>,
    result: &mut Vec<f32>,
    c: usize,
    batch_sz: usize,
) -> f64 {
    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;
    let mut analyzer: LRUSplay<(char, usize, usize, usize)> =
        LRUSplay::<(char, usize, usize, usize)>::new();

    let mut histogram_i = Hist::new();
    let mut histogram_k = Hist::new();
    let mut histogram_r = Hist::new();

    for p in 0..num_passes {
        for i in 0..(n - k + 1) {
            for j in 0..(n - k + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..k {
                        for x in 0..k {
                            let cur_k = analyzer.rec_access(('k', y, x, 0));
                            histogram_i.add_dist(cur_k);
                            let cur_i = analyzer.rec_access(('i', (i + y), (j + x), l));
                            histogram_k.add_dist(cur_i);
                        }
                    }
                }
                let cur_i = analyzer.rec_access(('r', i, j, 0));
                histogram_r.add_dist(cur_i);
            }
        }
    }

    let mut hist_vec = histogram_i.to_vec();
    let dmd_i = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });


    let mut hist_vec = histogram_k.to_vec();
    let dmd_k = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });

    let mut hist_vec = histogram_r.to_vec();
    let dmd_r = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });

    println!("n, {}, k, {}, c, {}, x, {}, dmd_i, {}, dmd_k, {}, dmd_r, {} ", n, k, c, batch_sz, dmd_i, dmd_k, dmd_r);


    dmd_i + dmd_k + dmd_r
}

pub fn conv_block(
    n: usize,
    k: usize,
    c: usize,
    batch_sz: usize,
    block_size: usize
) -> f64 {
    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;
    let mut analyzer: LRUSplay<(char, usize, usize, usize)> =
        LRUSplay::<(char, usize, usize, usize)>::new();

    let mut histogram_i = Hist::new();
    let mut histogram_k = Hist::new();
    let mut histogram_r = Hist::new();

//    println!("block size: {}, element size: {}, block cap {}", block_size, mem::size_of::<usize>(), block_size/mem::size_of::<usize>());

    for p in 0..num_passes {
        for i in 0..(n - k + 1) {
            for j in 0..(n - k + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..k {
                        for x in 0..k {
                            let cur_k = analyzer.rec_access(('k', y, x, 0));                            
                            histogram_i.add_dist(Some(cur_k.unwrap_or(0)/block_size));
                            let cur_i = analyzer.rec_access(('i', (i + y), (j + x), l));
                            histogram_k.add_dist(Some(cur_i.unwrap_or(0)/block_size));
                        }
                    }
                }
                let cur_r = analyzer.rec_access(('r', i, j, 0));
                histogram_r.add_dist(Some(cur_r.unwrap_or(0)/block_size));
            }
        }
    }

    let mut hist_vec = histogram_i.to_vec();
    let dmd_i = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });


    let mut hist_vec = histogram_k.to_vec();
    let dmd_k = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });

    let mut hist_vec = histogram_r.to_vec();
    let dmd_r = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });

    println!("n, {}, k, {}, c, {}, x, {}, block_size, {}, dmd_i, {}, dmd_k, {}, dmd_r, {}, dmd_total, {} ", n, k, c, batch_sz, block_size, dmd_i, dmd_k, dmd_r, dmd_i+dmd_k+dmd_r);


    dmd_i + dmd_k + dmd_r
}

pub fn fft_it(f: &mut Vec<Complex<f64>>, ln: usize, invert: bool) -> f64 {
    let n = 1 << ln;
    let half_n = n / 2;
    let mut j = 1;
    let mut analyzer = LRUSplay::<usize>::new();
    let mut dmc: f64 = 0.0;

    for i in 1..n {
        if i < j {
            f.swap(i - 1, j - 1);
            let mut cur = analyzer.rec_access(i - 1);
            dmc += (cur.unwrap_or(0) as f64).sqrt();
            cur = analyzer.rec_access(j - 1);
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
                let mut cur = analyzer.rec_access(m - 1);
                dmc += (cur.unwrap_or(0) as f64).sqrt();
                cur = analyzer.rec_access(i - 1);
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

pub fn fft_recursive(
    n: usize,
    mut a: Vec<Complex<f64>>,
    analyzer: &mut LRUSplay<(String, usize)>,
) -> Vec<Complex<f64>> {
    if n == 1 {
        a
    } else {
        unsafe {
            ITER += 1;
        }
        let iter: i32 = unsafe { ITER };
        let n_half = n / 2;

        let evens: Vec<Complex<f64>> = a.iter().step_by(2).cloned().collect();

        for i in 0..(evens.len()) {
            let a = format!("a{}", iter);
            let cur_e = analyzer.rec_access((a, i * 2));
            unsafe {
                GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();
            }
            //            let a_1 = format!("1{}", iter);
            //            let cur_e = analyzer.rec_access((1, i));
            //           unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
        }

        let mut f_even = fft_recursive(evens.len(), evens, analyzer);

        //        for i in 0..(f_even.len()) {
        //            let e_2 = format!("2{}", iter);
        //            let cur_e = analyzer.rec_access((e_2, i));
        //            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
        //            let e = format!("e{}", iter);
        //            let cur_o = analyzer.rec_access((e, i));
        //            unsafe { GLOBAL_DMC += (cur_o.unwrap_or(0) as f64).sqrt();}
        //       }
        let odds: Vec<Complex<f64>> = a.iter().skip(1).step_by(2).cloned().collect();

        for i in 0..odds.len() {
            let a = format!("a{}", iter);
            let cur_e = analyzer.rec_access((a, i * 2 + 1));
            unsafe {
                GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();
            }
            //            let a_3 = format!("3{}", iter);
            //            let cur_e = analyzer.rec_access((a_3, i));
            //            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
        }
        let mut f_odd = fft_recursive(odds.len(), odds, analyzer);

        for i in 0..f_odd.len() {
            //            let cur_e = analyzer.rec_access(("4".to_string(), i));
            //            unsafe { GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();}
            //            let a = format!("o{}", iter);
            //            let cur_o = analyzer.rec_access((a, i));
            //            unsafe { GLOBAL_DMC += (cur_o.unwrap_or(0) as f64).sqrt();}
        }

        for i in 0..f_even.len() {
            let cur_a = analyzer.rec_access((format!("a{}", iter), i));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }
            let a_next = format!("a{}", iter + 1);
            let cur_e = analyzer.rec_access((a_next, i));
            unsafe {
                GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();
            }
        }
        for i in 0..f_odd.len() {
            let cur_a = analyzer.rec_access((format!("a{}", iter), f_even.len() + i + 1));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }
            let a_next = format!("a{}", iter + 2);
            let cur_e = analyzer.rec_access((a_next, i));
            unsafe {
                GLOBAL_DMC += (cur_e.unwrap_or(0) as f64).sqrt();
            }
        }

        f_even.append(&mut f_odd);
        a = f_even;

        for k in 0..(n_half) {
            let w: Complex<f64> = Complex::<f64> {
                re: 0.,
                im: ((-(2.) * std::f64::consts::PI * k as f64) / n as f64),
            }
            .exp();
            let t = a[k];

            let mut cur_a = analyzer.rec_access((format!("a{}", iter), k));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }

            a[k] = t + w * a[k + (n / 2)];

            cur_a = analyzer.rec_access(("w".to_string(), k / n));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }
            cur_a = analyzer.rec_access((format!("a{}", iter), k + (n / 2)));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }
            cur_a = analyzer.rec_access((format!("a{}", iter), k));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }

            a[k + n / 2] = t - w * a[k + n / 2];

            cur_a = analyzer.rec_access(("w".to_string(), k / n));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }
            cur_a = analyzer.rec_access((format!("a{}", iter), k + (n / 2)));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }
            cur_a = analyzer.rec_access((format!("a{}", iter), k + (n / 2)));
            unsafe {
                GLOBAL_DMC += (cur_a.unwrap_or(0) as f64).sqrt();
            }
        }
        a
    }
}

pub fn matrix_multiplication(n_1: usize, n_2: usize, m_2: usize) -> f64 {

    let mut histogram = Hist::new();

    let mut analyzer: LRUSplay<(char, usize, usize)> = LRUSplay::<(char, usize, usize)>::new();

    for i in 0..n_1 {
        for j in 0..m_2 {
            for k in 0..n_2 {
                let cur_n = analyzer.rec_access(('n', i, k));
                histogram.add_dist(cur_n);

                let cur_m = analyzer.rec_access(('m', k, j));
                histogram.add_dist(cur_m);

                let cur_r = analyzer.rec_access(('r', i, j));
                histogram.add_dist(cur_r);
            }
        }
    }

    let mut hist_vec = histogram.to_vec();
    let dmd = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });

    dmd
}

pub fn matrix_multiplication_block(n_1: usize, n_2: usize, m_2: usize, block_size: usize) -> f64 {

    let mut histogram = Hist::new();

    let mut analyzer: LRUSplay<(char, usize, usize)> = LRUSplay::<(char, usize, usize)>::new();
    let block_cap = block_size/mem::size_of::<i32>();

    println!("block size: {}, element size: {}, block cap {}", block_size, mem::size_of::<i32>(), block_size/mem::size_of::<i32>());


    for i in 0..n_1 {
        for j in 0..m_2 {
            for k in 0..n_2 {
                let cur_n = analyzer.rec_access(('n', i, k));
                histogram.add_dist(Some(cur_n.unwrap_or(0)/block_cap));

                let cur_m = analyzer.rec_access(('m', k, j));
                histogram.add_dist(Some(cur_m.unwrap_or(0)/block_cap));

                let cur_r = analyzer.rec_access(('r', i, j));
                histogram.add_dist(Some(cur_r.unwrap_or(0)/block_cap));
            }
        }
    }

    let mut hist_vec = histogram.to_vec();
    let dmd = hist_vec.iter_mut().fold(0.0, |acc, (x, y)| {
        acc + ((*y as f64) * ((x.unwrap_or(0) as f64).sqrt())) as f64
    });

    dmd
}
