#![feature(linked_list_remove)]
use std::cmp::min;
use std::collections::LinkedList;
use std::env;

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

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3);

    let channels = args[1].parse::<usize>().unwrap();
    let batch_sz: i32 = args[2].parse::<i32>().unwrap();
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
    return ();
}
