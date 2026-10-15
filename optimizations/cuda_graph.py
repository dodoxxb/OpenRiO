"""Batch-one static CUDA execution. Each call aliases its reusable output buffers."""
import torch


class StaticForward:
    def __init__(self, model, device):
        self.input = torch.rand((1, 3, 800, 800), device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(10):
                model(self.input)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = model(self.input)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph.replay()
        torch.cuda.synchronize(device)

    def __call__(self, images):
        if images.shape != self.input.shape or images.dtype != torch.float32 or images.device != self.input.device:
            raise ValueError('Graph requires batch-one FP32 800x800 on its capture device')
        self.input.copy_(images)
        self.graph.replay()
        return self.output
