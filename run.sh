
min=$1
max=$2
for i in `seq $min $max`
do
    echo c = "$i" >> /home/ec2-user/results/batching/out_10_${max}
    /home/ec2-user/dev/convolution/target/release/convolution conv $i 10 >> /home/ec2-user/results/batching/out_10_${max}

#    first=`echo 2^$i |bc`
#    /home/ec2-user/convolution/target/release/convolution fft $first $i  >> /home/ec2-user/results/fft_2/out_${max}
done

