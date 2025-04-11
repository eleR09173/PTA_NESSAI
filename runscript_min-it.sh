#!/bin/bash
 


function runscript() {
    dt=$(date '+%Y-%m-%dT%T.%zZ')

    data_name=$(basename ${9}) # this is to add the last part of data_dir in the rundir name
 
    # Check if min_iteration is set or not. This is for the dir name
    if [ -n "${10}" ]; then
        iter_tag="minit${10}"
    else
        iter_tag="autoiter"
    fi

    # rundir=run_${dt}_${2}_${3}_${4}_${5}_${6}_${7}_${8}_${iter_tag} #remember that this way data_dir is not in the dir name. So if we accidentally run with the same parameters but different dataset, it will overwrite
    rundir=run_${dt}_${2}_${3}_${4}_${5}_${6}_${7}_${8}_${data_name}_${iter_tag} # this way the last part of data_dir is in the rundir name
    mkdir -p ${rundir}
    cd ${rundir}

    # Python command with the option min_iteration
    cmd="python $1 --nlive $2 --n_blocks $3 --n_layers $4 --n_neurons $5 --patience $6 --pytorch_threads $7 --n_pool $8 --data_dir $9"
    if [ -n "${10}" ]; then
        cmd="$cmd --min_iteration ${10}"
    fi

    echo "Running: $cmd"
    date '+%Y-%m-%dT%T.%zZ'
    eval $cmd
    date '+%Y-%m-%dT%T.%zZ'

}

 
homedir="/root/PTA_NESSAI"
cd ${homedir}
# runscript to select every time
# without min_iteration
# runscript ${homedir}/SimulatedPSR/MultiPSR/SimDR2newAllPSR_min-it.py 25000 10 10 40 20 6 6 ${homedir}/data/EPTA_DR2/DR2new
# NON SERVE AVERE DUE runscript DIVERSI. USO QUELLO CHE MIN_ITER E I $. DA TERMINALE METTO I NUMERI E LO SCRIPT LI METTE AL POSTO GIUSTO

# with min_iteration set
runscript ${homedir}/SimulatedPSR/MultiPSR/SimDR2newAllPSR_min-it.py $1 $2 $3 $4 $5 $6 $7 ${homedir}/data/EPTA_DR2/DR2new $8


# per la work station di Golam
# homedir="/homes/eleonora.villa/PTA-with-NessAI/PTA_NESSAI
###################################################### INTERACTIVE VERSION #####################################################################################
# #!/bin/bash

# # Ask for all arguments interactively - data_dir path /root/PTA_NESSAI/data/EPTA_DR2/DR2new
# read -p "Enter the number of live points (nlive): " nlive
# read -p "Enter the number of blocks (n_blocks): " n_blocks
# read -p "Enter the number of layers (n_layers): " n_layers
# read -p "Enter the number of neurons (n_neurons): " n_neurons
# read -p "Enter the patience (patience): " patience
# read -p "Enter the number of PyTorch threads (pytorch_threads): " pytorch_threads
# read -p "Enter the number of pools (n_pool): " n_pool
# read -p "Enter the data directory path (data_dir): " data_dir
# read -p "Enter the min_iteration value (leave empty for 'autoiter'): " min_iter

# # Check if a value for min_iteration was provided
# if [ -z "$min_iter" ]; then
#     echo "No value provided for min_iteration: using 'autoiter'"
#     min_iter="autoiter"
# fi

# # Get the current date and time to create a unique folder name
# dt=$(date '+%Y-%m-%dT%T.%zZ')

# # Extract the last part of the data directory path
# data_name=$(basename ${data_dir})

# # Use "autoiter" or the value of min_iter for the folder name
# if [ "$min_iter" == "autoiter" ]; then
#     iter_tag="autoiter"
# else
#     iter_tag="$min_iter"
# fi

# # Create the output directory name based on the provided arguments
# rundir=run_${dt}_${nlive}_${n_blocks}_${n_layers}_${n_neurons}_${patience}_${pytorch_threads}_${n_pool}_${data_name}_${iter_tag}

# # Create the directory and change into it
# mkdir -p ${rundir}
# cd ${rundir}

# # Create the command to execute based on whether min_iteration was provided or not
# homedir="/root/PTA_NESSAI"
# cd ${homedir}
# if [ "$min_iter" == "autoiter" ]; then
#     cmd="python ${homedir}/SimulatedPSR/MultiPSR/SimDR2newAllPSR.py $nlive $n_blocks $n_layers $n_neurons $patience $pytorch_threads $n_pool $data_dir"
# else
#     cmd="python ${homedir}/SimulatedPSR/MultiPSR/SimDR2newAllPSR.py $nlive $n_blocks $n_layers $n_neurons $patience $pytorch_threads $n_pool $data_dir $min_iter"
# fi

# # Execute the command
# echo "Running: $cmd"
# eval $cmd
