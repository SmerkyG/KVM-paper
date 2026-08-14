# Cluster Run

Topic hints: Verify process ownership before signaling ROCm jobs

## Lessons

- Before signaling a GPU process on a shared ROCm node, verify its full parent command and worktree against the intended cluster-run job; KFD/ROCm GPU indices may not match cluster-run allocation indices.
