
min=$1
max=$2
for i in `seq $min $max`
do
    echo c = "$i" >> /home/ec2-user/results/fft_2/out_${max}
#    /home/ec2-user/convolution/target/release/convolution conv $i 1 >> /home/ec2-user/results/one/out_${max}
#    /home/ec2-user/convolution/target/release/convolution fft 1048577 $i  >> /home/ec2-user/results/fft/out_${max}
    first=`echo 2^$i |bc`
    /home/ec2-user/convolution/target/release/convolution fft $first $i  >> /home/ec2-user/results/fft_2/out_${max}
done

