"""Small TP fixture: expandable cache plus registered, captured AITER AR.

Run with torchrun, two or more local gfx942 GPUs, and
PYTORCH_ALLOC_CONF=expandable_segments:True. No model weights are required.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from vllm_lod_plugin.graph_allocator import install_ipc_graph_allocator


def expandable(tensor):
    pointer = tensor.data_ptr()
    for segment in torch.cuda.memory._snapshot()["segments"]:
        if segment["address"] <= pointer < segment["address"] + segment["total_size"]:
            return segment["is_expandable"]
    raise AssertionError("tensor not found in allocator snapshot")


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    install_ipc_graph_allocator()
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    history = torch.ones(64 * 1024**2, device="cuda", dtype=torch.uint8)
    assert expandable(history)
    comm = CustomAllreduce(dist.group.WORLD, torch.cuda.current_device(), max_size=128 * 1024**2)
    assert not comm.disabled and comm.enable_register_for_capturing
    assert torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
    source = torch.full((8, 7168), rank + 1., device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), comm.capture():
        # Capture's warmup and allocations both stay inside the ordinary scope.
        comm.custom_all_reduce(source + 1)
        with torch.cuda.graph(graph, stream=stream):
            graph_input = source + 1
            output = comm.custom_all_reduce(graph_input)
        assert not expandable(graph_input)
        assert not expandable(output)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
    for change in [0, 3, 1]:
        source.fill_(rank + 1. + change)
        graph.replay()
        torch.cuda.synchronize()
        expected = world * (world + 1) / 2 + world * (1 + change)
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
    # Eager growth after capture remains expandable, not a global workaround.
    growth = torch.ones(128 * 1024**2, device="cuda", dtype=torch.uint8)
    assert expandable(growth)
    print(json.dumps(dict(rank=rank, world=world, registered_graph=True,
        graph_input_expandable=expandable(graph_input), history_expandable=expandable(history),
        growth_expandable=expandable(growth), correctness="passed")), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
