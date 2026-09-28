"""Device-neutral verification only; never changes inference or the intervention."""
import torch

def assert_actual_cast(actual, expected):
    # Native recorder may already have detached its endpoint to CPU.
    actual_cpu=actual.detach().cpu()
    expected_cpu=expected.detach().cpu().reshape_as(actual_cpu).to(actual_cpu.dtype)
    if not torch.equal(actual_cpu,expected_cpu):
        raise AssertionError('Actual native cast endpoint differs')
