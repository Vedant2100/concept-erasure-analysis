#!/usr/bin/env bash
# Master run script for the SPEED Reversibility Experiment
# Run this on your HPC cluster submit node.

set -e

echo "1. Submitting Environment Setup & Checkpoint Download..."
JOB_SETUP=$(sbatch --parsable experiments/slurm_setup.sh)
echo "   -> Job ID: $JOB_SETUP"

echo "2. Submitting Textual Inversion Recovery Probes (depends on setup)..."
JOB_PROBE=$(sbatch --parsable --dependency=afterok:$JOB_SETUP experiments/slurm_probe_ti.sh)
echo "   -> Job ID: $JOB_PROBE"

echo "3. Submitting Evaluation (depends on probes)..."
JOB_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_PROBE experiments/slurm_eval.sh)
echo "   -> Job ID: $JOB_EVAL"

echo ""
echo "All jobs queued! You can monitor them with 'squeue --me'."
echo "Once complete, results will be in:"
echo "  - results/probe_ti/snoopy/evaluation_metrics.csv"
echo "  - results/probe_ti/vangogh/evaluation_metrics.csv"
