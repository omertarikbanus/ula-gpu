"""Zero-copy Torch adapter for the per-robot controller."""
import ctypes as C
from contextlib import nullcontext
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
P, I = C.c_void_p, C.c_int


class BatchController:
    def __init__(self, n, device, library=None, backend='vendor'):
        self.n = n
        self.device = torch.empty(0, device=device).device
        self.cuda = self.device.type == 'cuda'
        if backend is None:
            backend = 'vendor'
        if not self.cuda and backend == 'inline':
            print('ula_gpu: backend "inline" ignored on CPU; the host path '
                  'always uses the built-in inline solver', file=sys.stderr)
        library = library or ROOT / ('build-cuda' if self.cuda else 'build') / 'libula_batch.so'
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.ula_batch_create.argtypes = [C.c_char_p, I, I, C.c_char_p]
        self.lib.ula_batch_create.restype = P
        self.lib.ula_batch_delete.argtypes = [P]
        self.lib.ula_batch_delete.restype = None
        self.lib.ula_batch_reset.argtypes = [P] * 5
        self.lib.ula_batch_tick.argtypes = [P] * 10 + [I, P]
        self.lib.ula_batch_act.argtypes = [P] * 5
        self.handle = None
        with self._guard():
            self.handle = self.lib.ula_batch_create(bytes(ROOT / 'models/go2/go2.urdf'), n, self.cuda,
                                                    backend.encode() if isinstance(backend, str) else backend)
        if not self.handle:
            raise RuntimeError('controller creation failed (see native error above)')

    def _guard(self):
        return torch.cuda.device(self.device) if self.cuda else nullcontext()

    def _ptr(self, a, width=None, dtype=torch.float32):
        shape = (self.n,) if width is None else (self.n, width)
        if a.shape != shape or not a.is_contiguous() or a.device != self.device or a.dtype != dtype:
            raise ValueError(f'expected contiguous {shape} {dtype} on {self.device}')
        if self.cuda:
            a.record_stream(torch.cuda.current_stream(self.device))
        return a.data_ptr()

    def _call(self, name, *args):
        if not self.handle:
            raise RuntimeError('controller has been closed')
        with self._guard():
            stream = torch.cuda.current_stream(self.device).cuda_stream if self.cuda else None
            err = getattr(self.lib, 'ula_batch_' + name)(self.handle, *args, stream)
        if err:
            raise RuntimeError(f'{name}: CUDA controller call failed ({err})')

    def reset(self, q, v, mask):
        self._call('reset', self._ptr(q, 19), self._ptr(v, 18), self._ptr(mask, dtype=torch.bool))

    def tick(self, q, v, action, command, noise, active, obs, vw, status, prime=False):
        self._call('tick', self._ptr(q, 19), self._ptr(v, 18), self._ptr(action, 25),
                   self._ptr(command, 3), self._ptr(noise, 9),
                   None if active is None else self._ptr(active, dtype=torch.bool),
                   self._ptr(obs, 54), self._ptr(vw, 4), self._ptr(status, dtype=torch.int32), prime)

    def act(self, length, velocity, ctrl):
        self._call('act', self._ptr(length, 12), self._ptr(velocity, 12), self._ptr(ctrl, 12))

    def close(self):
        if getattr(self, 'handle', None):
            with self._guard():
                self.lib.ula_batch_delete(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
