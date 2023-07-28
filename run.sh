
min=$1
max=$2
for i in `seq $min $max`
do
    echo c = "$i" >> /home/ec2-user/results/one/out_${max}
    /home/ec2-user/convolution/target/release/convolution $i 1 >> /home/ec2-user/results/one/out_${max}
done

