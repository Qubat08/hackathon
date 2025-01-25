#!/bin/sh
#SBATCH --job-name=job
#SBATCH --output=./output/R-%x.%j.out
#SBATCH --error=./error/R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --partition=c8g

# Load modules
module load acfl/24.10.1
module load gnu/14.2.0
module load armpl/24.10.1



# Compile the code  
make all
# all: BSM BSM1 BSM3-5 BSM6 BSM6 BSM6_arm BSM7 BSM8

# Run the code with different versions of the code 
# BSM
# echo "================EXECUTING BSM===================="
# OMP_NUM_THREADS=96 ./BSM 10000 1000000
# echo "================EXECUTING BSM1===================="
# OMP_NUM_THREADS=96 ./BSM1 100000 1000000
# echo "================EXECUTING BSM3-5===================="
# OMP_NUM_THREADS=96 ./BSM3-5 100000 1000000
# echo "================EXECUTING BSM6===================="
# OMP_NUM_THREADS=96 ./BSM6 100000 1000000
# echo "================EXECUTING BSM6_arm===================="
# OMP_NUM_THREADS=96 ./BSM6_arm 100000 1000000
# echo "================EXECUTING BSM7===================="
# OMP_NUM_THREADS=96 ./BSM7 100000 1000000
# echo "================EXECUTING BSM8===================="
# OMP_NUM_THREADS=96 ./BSM8 100000 1000000
# Faire ceci 9 fois pour les 9 versions du code

# echo "================EXECUTING BSM1===================="
# for i in {1..9}
# do
#     OMP_NUM_THREADS=96 ./BSM1 100000 1000000
# done

# echo "================EXECUTING BSM3-5===================="
# for i in {1..9}
# do
#     OMP_NUM_THREADS=96 ./BSM3-5 100000 1000000
# done

# echo "================EXECUTING BSM6===================="
# for i in {1..9}
# do
#     OMP_NUM_THREADS=96 ./BSM6 100000 1000000
# done

# echo "================EXECUTING BSM6_arm===================="
# for i in {1..9}
# do 
#     OMP_NUM_THREADS=96 ./BSM6_arm 100000 1000000
# done

# echo "================EXECUTING BSM7===================="
# for i in {1..9}
# do
#     OMP_NUM_THREADS=96 ./BSM7 100000 1000000
# done

# echo "================EXECUTING BSM8===================="
# for i in {1..9}
# do
#     OMP_NUM_THREADS=96 ./BSM8 100000 1000000
# done 

# Lancer la version la plus performante une fois en augmentant le premier argument de 1000 à 100000000 (fois 10)
echo "================EXECUTING BSM8===================="
i=1000
while [ $i -le 100000000 ]
do
    OMP_NUM_THREADS=96 ./BSM8 $i 1000000
    i=$((i * 10))
done



# End of file