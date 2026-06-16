"""Progress reporter for the JellyFish NS solver.

Detects SLURM vs interactive sessions and switches between a tqdm progress
bar (interactive) and plain periodic prints (SLURM/batch).
"""

import os
import sys
import numpy as np

_IN_SLURM = 'SLURM_JOB_ID' in os.environ

HEADER = """\
Navier Stokes solver on DOLFINx
"""

_NORM_FMT = '  |u_t|={:.3e}  |p|={:.3e}  |u|={:.3e}'



class ProgressReporter:
    """Wraps the time loop with a progress bar or batch-friendly prints.

    Args:
        total_steps (int):  number of time steps
        dt (float):         time step size
        rank (int):         MPI rank — only rank 0 prints
        report_interval (int): steps between norm reports (0 = auto ~20)
    """

    def __init__(self, total_steps, dt, rank=0, report_interval=0):
        self.total = total_steps
        self.dt = dt
        self.rank = rank
        self._slurm = _IN_SLURM

        if report_interval <= 0:
            self._interval = max(1, total_steps // 20)
        else:
            self._interval = report_interval

        self._bar = None
        self._norm_u_tent = float('nan')
        self._norm_p = float('nan')
        self._norm_u_upd = float('nan')

    # ------------------------------------------------------------------
    def print_banner(self):
        if self.rank != 0:
            return

        lines = [HEADER, '']
        msg = '\n'.join(lines)
        
        if self._slurm or self._bar is None:
            print(msg, flush=True)
        else:
            from tqdm import tqdm as _tqdm
            _tqdm.write(msg)

    def start(self):
        """Open the progress bar (interactive only)."""
        if self.rank != 0 or self._slurm:
            return
        try:
            from tqdm import tqdm
        except ImportError:
            return
        self._bar = tqdm(
            total=self.total,
            unit='step',
            bar_format=(
                '{l_bar}{bar}| {n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}]{postfix}'
            ),
            file=sys.stdout,
            dynamic_ncols=True,
        )

    def update(self, step, t, norm_u_tent=None, norm_p=None, norm_u_upd=None):
        """Advance one step and optionally report norms."""
        if norm_u_tent is not None:
            self._norm_u_tent = norm_u_tent
        if norm_p is not None:
            self._norm_p = norm_p
        if norm_u_upd is not None:
            self._norm_u_upd = norm_u_upd

        should_report = (step % self._interval == 0) or (step == self.total)

        if self.rank != 0:
            return

        if self._bar is not None:

            #self._bar.set_postfix_str(
            #    '|u_t|={:.2e} |p|={:.2e} |u|={:.2e}'.format(
            #        self._norm_u_tent, self._norm_p, self._norm_u_upd),
            #    refresh=False,
            #)
            self._bar.update(1)

            if should_report:
                from tqdm import tqdm as _tqdm
                _tqdm.write(
                    'step {:5d}  t={:.4f}'.format(step, t) +
                    _NORM_FMT.format(self._norm_u_tent, self._norm_p,
                                     self._norm_u_upd)
                )
        else:
            # SLURM / fallback: one line per step + periodic norm line
            if should_report:
                print(
                    'step {:5d}  t={:.4f}'.format(step, t) +
                    _NORM_FMT.format(self._norm_u_tent, self._norm_p,
                                     self._norm_u_upd),
                    flush=True,
                )

    def close(self):
        if self._bar is not None and self.rank == 0:
            self._bar.close()
            self._bar = None
