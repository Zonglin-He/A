"""One persistent, isolated ReferDINO; only RGB and caption cross the boundary."""
import multiprocessing
import time


def worker(connection, log_path):
    import contextlib
    import gc
    import traceback
    import torch
    from .runtime import setup
    from .teacher_worker import deny_detector
    import sys
    setup()
    sys.addaudithook(deny_detector)
    with open(log_path, 'a', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            from .teacher import SingleRefer
            tick = time.perf_counter()
            teacher = SingleRefer()
            teacher.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            load_seconds = time.perf_counter()-tick
            connection.send(dict(status='ready', load_seconds=load_seconds))
            while True:
                request = connection.recv()
                if request is None:
                    break
                if set(request) != {'frames', 'caption'}:
                    raise ValueError('Teacher accepts only frames/caption')
                torch.manual_seed(42)
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                teacher.model.cuda()
                value = teacher(request['frames'], request['caption'])
                peak = torch.cuda.max_memory_allocated()
                teacher.model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                value.update(model_load_seconds=load_seconds, peak_GPU_bytes=peak,
                             process_wall_seconds=time.perf_counter()-tick,
                             load_audit=teacher.load_audit, persistent_singleton=True)
                # Use plain arrays over the process boundary, no CUDA/shared tensor handles.
                value = {k: v.numpy() if torch.is_tensor(v) else v for k, v in value.items()}
                connection.send(dict(status='ok', value=value))
        except BaseException:
            connection.send(dict(status='error', error=traceback.format_exc()))
        finally:
            connection.close()


class PersistentRefer:
    def __init__(self, log_path):
        from .runtime import ROOT, sha
        import json
        path = 'checkpoints/referdino/ryt_mevis_swinb.pth'
        expected = json.loads((ROOT/'docs/assets.json').read_text())['required_files'][path]
        if sha(ROOT/path) != expected:
            raise RuntimeError('ReferDINO checkpoint fingerprint differs from the locked expert')
        ctx = multiprocessing.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=worker, args=(child, str(log_path)))
        self.process.start()
        child.close()
        try:
            reply = self._receive(240)
            if reply['status'] != 'ready':
                raise RuntimeError(reply)
        except BaseException:
            self.close()
            raise
        self.load_seconds = reply['load_seconds']

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError('ReferDINO worker did not reply')
        reply = self.connection.recv()
        if reply['status'] == 'error':
            raise RuntimeError(reply['error'])
        return reply

    def __call__(self, frames, caption):
        self.connection.send(dict(frames=frames, caption=caption))
        return self._receive(240)['value']

    def close(self):
        if self.process.is_alive():
            self.connection.send(None)
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()  # only this explicitly owned worker
            self.process.join()
        self.connection.close()
