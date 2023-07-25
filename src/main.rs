#![feature(linked_list_remove)]   // Need nightly build of the Rust compiler
use std::cmp::min;
//use std::f64::sqrt;
use std::env;
use std::collections::LinkedList;

const n: usize = 4;
const k: usize = 2;

static mut GLOBAL_DMC: f64 = 0.0;


unsafe fn update_dmc(sampler: &mut LinkedList<u64>, addr_ptr: *const f32) {
    let mut addr = addr_ptr as u64;

    let mut pos:usize = 0;
    let mut it_prev = sampler.iter_mut().peekable();
    while let Some(val) = it_prev.next() {
        if *val == addr {
            it_prev.next();
            GLOBAL_DMC += (pos as f64).sqrt() ;
    println!("sub: {}", pos );
    break;
        }
        pos += 1;
    }
    if pos != sampler.len() {
        sampler.remove(pos as usize);
    }
    sampler.push_front(addr);
}


fn conv2d(I: &mut [f32], K: &mut [f32], R: &mut [f32], c: usize, batch_sz: usize) {
    let num_passes = (c as f64 / batch_sz as f64).ceil() as usize;
    let mut sampler: LinkedList<u64> = LinkedList::new();

    for p in 0..num_passes {
        for i in 0..(n - k + 1) {
            for j in 0..(n - k + 1) {
                let mut sum = 0.0;
                for l in (p * batch_sz)..min((p + 1) * batch_sz, c) {
                    for y in 0..k {
                        for x in 0..k {
                            sum += K[y * k + x] * I[((i + y) * n + (j + x)) * c + l];
                            unsafe {
                                println!("ptr: K[{}]", y * k + x );
                                update_dmc(&mut sampler, &K[y * k + x]);
                                println!("ptr: I[{}]", ((i + y) * n + (j + x)) * c + l );
                                update_dmc(&mut sampler, &I[((i + y) * n + (j + x)) * c + l]);
                            }
                        }
                    }
                }
                R[i * n + j] += sum;
                unsafe {
                    println!("ptr: R[{}]", i * n + j);
                    update_dmc(&mut sampler, &R[i * n + j]);
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3);

    let channels = args[1].parse::<usize>().unwrap();
    let batch_sz: i32 = args[2].parse::<i32>().unwrap();
    let batch_sz = if batch_sz == -1 { channels as i32 } else { batch_sz as i32 };

    let mut I = vec![0.0; n * n * channels];
    let mut R = vec![0.0; n * n];
    let mut K = vec![0.0; k * k];
    conv2d(&mut I, &mut K, &mut R, channels, batch_sz as usize);
    println!("DMC: {}", unsafe { GLOBAL_DMC });
    return ();// if R[0] < 0.0 { 1 } else { 0 };
}
