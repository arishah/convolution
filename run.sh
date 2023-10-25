
min=$1
max=$2
n=$3
x=$4
for i in `seq $min $max`
do
    echo n = "$i" >> /home/ec2-user/results/conv_n/out_${n}_${x}_${max}
    /home/ec2-user/dev/convolution/target/release/convolution conv $i 3 $n $x >> /home/ec2-user/results/conv_n/out_${n}_${x}_${max}

#    first=`echo 2^$i |bc`
#    /home/ec2-user/dev/convolution/target/release/convolution fft $first $i  >> /home/ec2-user/results/fft_2/out_${max}

#    /home/ec2-user/dev/convolution/target/release/convolution alg $i 3 $n $x >> /home/ec2-user/results/conv_time/out_${n}_${x}_${max}

done

