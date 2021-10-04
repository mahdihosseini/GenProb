#Define list of hyperparameter varations

default = '''#!/bin/bash 
### GPU OPTIONS:
### CEDAR: v100l, p100
### BELUGA: *no option, just use --gres=gpu:*COUNT*
### GRAHAM: v100, t4
### see https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --account=def-plato
#SBATCH --time=24:00:0
###SBATCH --constraint=cascade

source ~/scratch/June/QC-Bench/QC-BenchEnv/bin/activate

'''
i = 1
nums = ['0', '10', '20', '30', '40','50', '60', '70']
for a in range(7):
    f = open("batch_files/main" + str(a) + ".sh", "w")
    f.write(default)
    
    start  = nums[a]
    finish = nums[a + 1]

    print(i, ": python ~/scratch/June/QC-Bench/source/main.py " + start + " " + finish)
    i += 1

    f.write("python ~/scratch/June/QC-Bench/source/main.py " + start + " " + finish)
    f.write("\n")
    f.close()