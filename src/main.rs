use num_complex::Complex;
use rand::Rng;
use stack_alg_sim::olken::LRUSplay;
use stack_alg_sim::LRU;
use std::cmp::min;
use std::env;
use std::f64::consts::PI;
use hist::Hist;
mod algorithm;
use algorithm::*;
mod simulation;
use simulation::*;

//const N: usize = 128;
//const K: usize = 3;

static mut GLOBAL_DMC: f64 = 0.0;
static mut ITER: i32 = 0;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args[1].parse::<String>().unwrap();

    if mode == "conv" {
        let n = args[2].parse::<usize>().unwrap();
        let k = args[3].parse::<usize>().unwrap();
        let channels = args[4].parse::<usize>().unwrap();
        let batch_sz: i32 = args[5].parse::<i32>().unwrap();
        let batch_sz = if batch_sz == -1 {
            channels as i32
        } else {
            batch_sz as i32
        };

        //        let mut image_i = vec![0.0; n * n * channels];
        let mut image_i = vec![0.0; n * n];
        let mut result = vec![0.0; n * n];
        let mut image_k = vec![0.0; k * k];

        let dmc = conv(
            n,
            //            &mut image_i,
            image_i,
            k,
            //            &mut image_k,
            image_k,
            &mut result,
            channels,
            batch_sz as usize,
        );
//        println!("DMC: {}", dmc);
    } else if mode == "conv-block" {
        let n = args[2].parse::<usize>().unwrap();
        let k = args[3].parse::<usize>().unwrap();
        let block_size = args[4].parse::<usize>().unwrap_or(64);
        let channels = args[5].parse::<usize>().unwrap();
        let batch_sz: i32 = args[6].parse::<i32>().unwrap();
        let batch_sz = if batch_sz == -1 {
            channels as i32
        } else {
            batch_sz as i32
        };

        let block_size :usize = 64;

        let dmc = conv_block(
            n,
            k,
            channels,
            batch_sz as usize,
            block_size
        );
    } else if mode == "fft" {
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

        let mut analyzer: LRUSplay<(String, usize)> = LRUSplay::<(String, usize)>::new();
        fft_recursive(a.len(), a, &mut analyzer);
        println!("DMC: {}", unsafe { GLOBAL_DMC });
    } else if mode == "conv_alg" {
        let n = args[2].parse::<usize>().unwrap();
        let k = args[3].parse::<usize>().unwrap();
        let channels = args[4].parse::<usize>().unwrap();
        let batch_sz: i32 = args[5].parse::<i32>().unwrap();
        let batch_sz = if batch_sz == -1 {
            channels as i32
        } else {
            batch_sz as i32
        };

        let mut rng = rand::thread_rng();
        let mut image_i: Vec<Vec<f64>> = (0..n*channels).map(|_| {
            (0..n*channels)
            .map(|_| {
                let im: f64 = rng.gen_range(0.0..25.0);
                im
            })
            .collect()
        }).collect();

        let mut image_k: Vec<Vec<f64>> = (0..k).map(|_| {
            (0..k)
            .map(|_| {
                let im: f64 = rng.gen_range(0.0..25.0);
                im
            })
            .collect()
        }).collect();


/*
        let mut image_i: Vec<f32> = vec![6.103921, 18.088875, 24.588144, 4.452592, 8.66113, 18.30811, 12.927482, 7.5458345, 7.543391, 12.42911, 6.1582804, 9.538987, 14.501134, 19.025675, 10.325516, 15.668571, 7.245472, 6.7605405, 13.725066, 0.37198067, 20.564487, 0.71589947, 19.81746, 4.977572, 17.874912, 24.104904, 19.471487, 22.58652, 15.882486, 12.089491, 4.7557592, 18.325645, 18.374428, 10.757292, 7.985902, 11.966094, 15.375721, 17.771816, 24.581543, 3.776929, 8.7210655, 17.177826, 13.728529, 5.3876133, 3.0119987, 22.473898, 4.266429, 11.485407, 3.6971302, 1.3400793, 20.758013, 17.659027, 15.115828, 3.1386793, 15.256229, 21.415401, 14.749455, 21.25045, 5.40705, 15.212935, 3.254068, 16.391167, 9.574974, 18.995726, 7.639706, 24.372646, 9.4953985, 21.21566, 17.071888, 19.780327, 16.541893, 1.6696095, 6.9995165, 2.0852594, 21.380524, 23.546501, 7.3394446, 7.0526896, 15.451416, 5.2989125, 24.805737, 5.1555276, 4.7283144, 6.273803, 6.3921185, 10.637063, 3.4396677, 7.243514, 7.4520917, 1.7634153, 13.5028515, 9.3214035, 10.610342, 9.907881, 6.314692, 13.0663, 2.2519946, 1.3910443, 17.301708, 22.36745, 4.838443, 22.942616, 16.93202, 13.80162, 4.878214, 24.663895, 20.01187, 15.431317, 10.648963, 6.8926277, 17.156904, 13.353857, 4.793182, 18.509579, 2.954218, 2.7701676, 11.386278, 18.488153, 15.677118, 0.6226301, 15.088153, 19.581848, 13.034323, 4.3180137, 22.302998, 2.0891786, 3.8186371, 18.503452, 20.51631, 22.926495, 10.699311, 6.5155954, 6.600511, 13.007728, 20.302275, 7.2060676, 14.700887, 16.8899, 13.75655, 22.35949, 3.3265023, 10.8144045, 23.009249, 19.233671, 3.7878394, 1.4235854, 13.114971, 7.52615, 16.159122, 13.821721, 7.8401685, 3.6476283, 12.200337, 15.791813, 14.37138, 16.83721, 23.585527, 24.017488, 20.890158, 9.04831, 7.134223, 8.06728, 21.990171, 20.491346, 10.990852, 12.439808, 8.103455, 9.600788, 9.234524, 16.640265, 14.533591, 16.45421, 4.0856214, 21.991562, 1.1784762, 15.642372, 2.376947, 14.071316, 15.962559, 9.51018, 21.221212, 1.0926694, 4.6896753, 17.914883, 23.525433, 1.70843, 3.480196, 7.270828, 4.6140404, 6.2529325, 6.646624, 10.2689295, 19.918045, 6.8282366, 7.6397867, 5.5226984, 21.326416, 13.092965, 10.111019, 4.92641, 7.236928, 23.608118, 11.068937, 23.116693, 2.7492583, 15.815822, 18.415964, 24.473106, 16.328218, 16.585716, 0.38232803, 9.678682, 9.671193, 5.7083426, 16.089293, 22.761423, 0.55247843, 5.421501, 0.44960976, 19.979733, 5.6622744, 17.95765, 17.919552, 14.725152, 18.58947, 16.54581, 17.492693, 4.6750603, 8.710608, 0.10488331, 8.751572, 16.410011, 14.053288, 0.62037706, 2.5430174, 23.624393, 11.618879, 22.263607, 19.888746, 19.729027, 13.93013, 4.6450796, 16.86132, 11.601583, 9.299183, 5.856809, 20.121298, 20.051872, 22.55596, 8.549762, 24.131207, 7.4764194, 7.2917013, 19.1894, 19.179144, 19.736168, 19.049784, 13.10438, 3.9213033, 5.1112204, 6.4954014, 10.360047, 13.616312, 14.718476, 0.1829952, 24.854177, 9.4269905, 9.902999, 15.085095, 19.613998, 15.866125, 13.923466, 16.796707, 7.844913, 8.110619, 2.262786, 0.44938326, 17.311405, 19.650272, 15.890842, 11.645579, 4.746169, 0.73584914, 18.669525, 0.21314025, 3.048143, 2.3293555, 11.139697, 9.741318, 18.377485, 22.789398, 12.004984, 14.842948, 14.902202, 9.777987, 12.112323, 16.925392, 19.011152, 17.4328, 7.7059984, 8.994767, 3.1927884, 3.7446141, 5.29508, 19.386017, 18.438711, 5.256066, 17.4939, 12.848004, 8.366487, 21.284178, 13.880745, 6.956607, 7.9702883, 13.834742, 4.7498107, 10.56534, 21.762857, 4.311246, 15.030914, 15.346593, 22.349289, 4.4058027, 9.171978, 21.0767, 7.4356375, 1.22599, 7.7827873, 23.400778, 18.280634, 8.407471, 17.833609, 13.29678, 8.443848, 1.0921001, 13.348937, 3.2348423, 0.38388968, 2.9419212, 24.074762, 16.578123, 24.715603, 24.369503, 7.9167037, 24.727463, 10.096807, 22.01348, 8.341831, 9.661126, 8.780798, 16.369291, 12.322962, 21.09938, 20.228794, 24.020704, 1.255402, 10.252336, 2.3138971, 12.379649, 24.163803, 12.820411, 5.546117, 5.862525, 16.252514, 2.3257256, 16.183346, 18.997469, 19.702255, 20.060581, 23.11534, 19.076023, 24.766958, 12.527171, 19.800726, 23.690924, 4.2699485, 3.9622812, 7.9161825, 16.699568, 12.98703, 22.340939, 14.594051, 21.740469, 0.7596761, 5.141151, 3.5127342, 24.577215, 5.941424, 20.913786, 16.135412, 19.458315, 19.39443, 20.901028, 7.845873, 6.7642956, 19.093475, 21.411272, 24.767376, 23.86587, 22.506355];
        let mut image_k: Vec<f32> = vec![18.059418, 13.774434, 1.8028915, 9.967682, 10.281944, 2.4214, 3.6973119, 24.533415, 24.544514];


let mut image_i: Vec<Vec<f64>> = vec![vec![6.103921, 18.088875, 24.588144, 4.452592, 8.66113, 18.30811, 12.927482, 7.5458345, 7.543391, 12.42911, 6.1582804, 9.538987, 14.501134, 19.025675, 10.325516, 15.668571, 7.245472, 6.7605405, 13.725066, 0.37198067], 
                                    vec![20.564487, 0.71589947, 19.81746, 4.977572, 17.874912, 24.104904, 19.471487, 22.58652, 15.882486, 12.089491, 4.7557592, 18.325645, 18.374428, 10.757292, 7.985902, 11.966094, 15.375721, 17.771816, 24.581543, 3.776929], 
                                    vec![8.7210655, 17.177826, 13.728529, 5.3876133, 3.0119987, 22.473898, 4.266429, 11.485407, 3.6971302, 1.3400793, 20.758013, 17.659027, 15.115828, 3.1386793, 15.256229, 21.415401, 14.749455, 21.25045, 5.40705, 15.212935], 
                                    vec![3.254068, 16.391167, 9.574974, 18.995726, 7.639706, 24.372646, 9.4953985, 21.21566, 17.071888, 19.780327, 16.541893, 1.6696095, 6.9995165, 2.0852594, 21.380524, 23.546501, 7.3394446, 7.0526896, 15.451416, 5.2989125], 
                                    vec![24.805737, 5.1555276, 4.7283144, 6.273803, 6.3921185, 10.637063, 3.4396677, 7.243514, 7.4520917, 1.7634153, 13.5028515, 9.3214035, 10.610342, 9.907881, 6.314692, 13.0663, 2.2519946, 1.3910443, 17.301708, 22.36745], 
                                    vec![4.838443, 22.942616, 16.93202, 13.80162, 4.878214, 24.663895, 20.01187, 15.431317, 10.648963, 6.8926277, 17.156904, 13.353857, 4.793182, 18.509579, 2.954218, 2.7701676, 11.386278, 18.488153, 15.677118, 0.6226301], 
                                    vec![15.088153, 19.581848, 13.034323, 4.3180137, 22.302998, 2.0891786, 3.8186371, 18.503452, 20.51631, 22.926495, 10.699311, 6.5155954, 6.600511, 13.007728, 20.302275, 7.2060676, 14.700887, 16.8899, 13.75655, 22.35949], 
                                    vec![3.3265023, 10.8144045, 23.009249, 19.233671, 3.7878394, 1.4235854, 13.114971, 7.52615, 16.159122, 13.821721, 7.8401685, 3.6476283, 12.200337, 15.791813, 14.37138, 16.83721, 23.585527, 24.017488, 20.890158, 9.04831], 
                                    vec![7.134223, 8.06728, 21.990171, 20.491346, 10.990852, 12.439808, 8.103455, 9.600788, 9.234524, 16.640265, 14.533591, 16.45421, 4.0856214, 21.991562, 1.1784762, 15.642372, 2.376947, 14.071316, 15.962559, 9.51018], 
                                    vec![21.221212, 1.0926694, 4.6896753, 17.914883, 23.525433, 1.70843, 3.480196, 7.270828, 4.6140404, 6.2529325, 6.646624, 10.2689295, 19.918045, 6.8282366, 7.6397867, 5.5226984, 21.326416, 13.092965, 10.111019, 4.92641], 
                                    vec![7.236928, 23.608118, 11.068937, 23.116693, 2.7492583, 15.815822, 18.415964, 24.473106, 16.328218, 16.585716, 0.38232803, 9.678682, 9.671193, 5.7083426, 16.089293, 22.761423, 0.55247843, 5.421501, 0.44960976, 19.979733], 
                                    vec![5.6622744, 17.95765, 17.919552, 14.725152, 18.58947, 16.54581, 17.492693, 4.6750603, 8.710608, 0.10488331, 8.751572, 16.410011, 14.053288, 0.62037706, 2.5430174, 23.624393, 11.618879, 22.263607, 19.888746, 19.729027], 
                                    vec![13.93013, 4.6450796, 16.86132, 11.601583, 9.299183, 5.856809, 20.121298, 20.051872, 22.55596, 8.549762, 24.131207, 7.4764194, 7.2917013, 19.1894, 19.179144, 19.736168, 19.049784, 13.10438, 3.9213033, 5.1112204], 
                                    vec![6.4954014, 10.360047, 13.616312, 14.718476, 0.1829952, 24.854177, 9.4269905, 9.902999, 15.085095, 19.613998, 15.866125, 13.923466, 16.796707, 7.844913, 8.110619, 2.262786, 0.44938326, 17.311405, 19.650272, 15.890842], 
                                    vec![11.645579, 4.746169, 0.73584914, 18.669525, 0.21314025, 3.048143, 2.3293555, 11.139697, 9.741318, 18.377485, 22.789398, 12.004984, 14.842948, 14.902202, 9.777987, 12.112323, 16.925392, 19.011152, 17.4328, 7.7059984], 
                                    vec![8.994767, 3.1927884, 3.7446141, 5.29508, 19.386017, 18.438711, 5.256066, 17.4939, 12.848004, 8.366487, 21.284178, 13.880745, 6.956607, 7.9702883, 13.834742, 4.7498107, 10.56534, 21.762857, 4.311246, 15.030914], 
                                    vec![15.346593, 22.349289, 4.4058027, 9.171978, 21.0767, 7.4356375, 1.22599, 7.7827873, 23.400778, 18.280634, 8.407471, 17.833609, 13.29678, 8.443848, 1.0921001, 13.348937, 3.2348423, 0.38388968, 2.9419212, 24.074762], 
                                    vec![16.578123, 24.715603, 24.369503, 7.9167037, 24.727463, 10.096807, 22.01348, 8.341831, 9.661126, 8.780798, 16.369291, 12.322962, 21.09938, 20.228794, 24.020704, 1.255402, 10.252336, 2.3138971, 12.379649, 24.163803], 
                                    vec![12.820411, 5.546117, 5.862525, 16.252514, 2.3257256, 16.183346, 18.997469, 19.702255, 20.060581, 23.11534, 19.076023, 24.766958, 12.527171, 19.800726, 23.690924, 4.2699485, 3.9622812, 7.9161825, 16.699568, 12.98703], 
                                    vec![22.340939, 14.594051, 21.740469, 0.7596761, 5.141151, 3.5127342, 24.577215, 5.941424, 20.913786, 16.135412, 19.458315, 19.39443, 20.901028, 7.845873, 6.7642956, 19.093475, 21.411272, 24.767376, 23.86587, 22.506355]];

let mut image_k: Vec<Vec<f64>> = vec![vec![18.059418, 13.774434, 1.8028915], 
                                    vec![9.967682, 10.281944, 2.4214], 
                                    vec![3.6973119, 24.533415, 24.544514]];
*/



//        let mut result = vec![vec![0.0; n]; n];
        let result = 
        convolution(
            n,
            image_i,
            k,
            image_k,
//            &mut result,
            channels,
            batch_sz as usize,
        );
    } else if mode == "mm_alg" {
        let n_1 = args[2].parse::<usize>().unwrap();
        let n_2 = args[3].parse::<usize>().unwrap();
        let m_2 = args[4].parse::<usize>().unwrap();
        

        let mut rng = rand::thread_rng();
        let mut n: Vec<Vec<i32>> = (0..n_1).map(|_| {
            (0..n_2)
            .map(|_| {
                let im: i32 = rng.gen_range(0..25);
                im
            })
            .collect()
        }).collect();

        let mut m: Vec<Vec<i32>> = (0..n_2).map(|_| {
            (0..m_2)
            .map(|_| {
                let im: i32 = rng.gen_range(0..25);
                im
            })
            .collect()
        }).collect();

        let result = matrix_multi(n, m);
        for row in &result {
            println!("{:?}", row);
        }
    } else if mode == "mm" {
        let n_1 = args[2].parse::<usize>().unwrap();
        let n_2 = args[3].parse::<usize>().unwrap();
        let m_2 = args[4].parse::<usize>().unwrap();
        

        let dmd = matrix_multiplication(n_1, n_2, m_2);
        println!("n_rows, {}, n_columns, {}, m_rows, {}, DMD, {}", n_1, n_2, m_2, dmd);
    } else if mode == "mm_block" {
        let n_1 = args[2].parse::<usize>().unwrap();
        let n_2 = args[3].parse::<usize>().unwrap();
        let m_2 = args[4].parse::<usize>().unwrap();
        let block_size = args[5].parse::<usize>().unwrap();
        
        let dmd = matrix_multiplication_block(n_1, n_2, m_2, block_size);
        println!("n_rows, {}, n_columns, {}, m_rows, {}, block_size, {}, DMD, {}", n_1, n_2, m_2, block_size, dmd);
    }
    return ();
}
