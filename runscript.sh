#!/bin/bash
 


function runscript() {
    dt=$(date '+%Y-%m-%dT%T.%zZ')
    rundir=run_${dt}_${2}_${3}_${4}_${5}_${6}_${7}_${8}
    mkdir -p ${rundir}
    cd ${rundir}
    python $1 --nlive $2 --n_blocks $3 --n_layers $4 --n_neurons $5 --patience $6 --pytorch_threads $7 --n_pool $8 --data_dir $9
}


homedir="/root/PTA_NESSAI"
cd ${homedir}
runscript ${homedir}/SimulatedPSR/MultiPSR/SimDR2newAllPSR.py 25000 10 10 40 20 6 6 ${homedir}/data/EPTA_DR2/DR2new



 
 