"""``cd task_3 && python -m pa4t3.run_all [--smoke] [--gpus 0,1]``: the whole of Task 3 through the resumable scheduler."""
import sys

from pa4common.scheduler import cli

from .jobs import full_jobs, smoke_jobs

if __name__ == "__main__":
    sys.exit(cli(full_jobs, smoke_jobs))
