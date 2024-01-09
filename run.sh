
min=$1
max=$2
n=$3
x=$4
for i in `seq $min 10 $max`
do
#    echo n = "$i" >> /u/ashahmir/DMC/results/conv_n/out_${n}_${x}_${max}
#    /u/ashahmir/DMC/convolution/target/release/convolution conv $n 10 $i $x >> /u/ashahmir/DMC/results/conv_n/out_${n}_${x}_${max}

#    first=`echo 2^$i |bc`
#    /home/ec2-user/dev/convolution/target/release/convolution fft $first $i  >> /home/ec2-user/results/fft_2/out_${max}

    /u/ashahmir/DMC/convolution/target/release/convolution alg $n 100 $i $x >> /u/ashahmir/DMC/results/conv_time/out_${n}_${x}_${max}

done

