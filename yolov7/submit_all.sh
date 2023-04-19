#!/bin/bash
# submit all slurm jobs for attack testing
ls | sed -n 's/\(test_tiny.*.slurm\)/sbatch \1/p'| bash