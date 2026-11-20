Benchmark reports, plots, submission records, source hashes, and compact result
files are versioned here. The AMD/H100 integration snapshot excludes duplicate
source archives, cached full prompts, tensor captures, execution logs, and other
individual result files larger than 1 MiB. These files remain in the original
experiment workspaces; this commit does not delete them.

The source and cohort manifests retained alongside the reports identify the
frozen code and benchmark inputs used for the runs. Large inputs can be
reconstructed with the preparation commands recorded by each experiment.
