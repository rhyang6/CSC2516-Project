#!/bin/bash
# submit all slurm jobs for attack testing
ls | sed -n 's/\(test.*\_freeze.slurm\)/sbatch \1/p'| bash