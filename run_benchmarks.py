import sys
import os
from pathlib import Path
from typing import List
import contextlib
import subprocess
from warnings import warn

REPO_ROOT = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results"


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class BenchmarkJobConfig:

    def __init__(self, repo_name: str, path_to_benchmark_job_relative: Path):
        self.repo_name: str = repo_name
        self.repo_dir: Path = REPO_ROOT / self.repo_name

        if not self.repo_dir.exists():
            raise FileNotFoundError("Repo directory doesn't exist. Make sure you initialized the "
                                    "GPU profiling repo.")

        # Python file to run the benchmark.
        self.benchmark_path: Path = self.repo_dir / path_to_benchmark_job_relative
        if not self.benchmark_path.exists():
            raise FileNotFoundError("Couldn't find benchmark job with path:", self.benchmark_path)


def execute(cmd: List[str]):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_job(job_config: BenchmarkJobConfig, python_executable: str, rig_results_dir: Path):
    print(f"Running {job_config.repo_name} benchmarks")
    outputs = []
    result_file = (rig_results_dir / f"gpu_profiling_{job_config.repo_name}.txt").resolve()

    # Since pytorch_kinematics expects the current working directory to be the directory the tests
    # are located in, we have to change that temporarily.
    with working_directory(job_config.benchmark_path.parent):
        # This looks strange but it's done so that we can print the job's STDOUT messages while also
        # capturing the output (so that we can store the results). The "-u" flag tells Python to not
        # buffer any messages printed to STDOUT.
        cmd = [python_executable, "-u", job_config.benchmark_path.name]
        for line in execute(cmd):
            outputs.append(line)
            print(line, end='', flush=True)
    print("Finished running job.")

    # Save job's output.
    output_str = "".join(outputs)
    with result_file.open('w') as f:
        f.write(output_str)
    print("Wrote job results to:", result_file)


def get_hostname():
    ret = subprocess.run(["hostname"], stdout=subprocess.PIPE)
    hostname = ret.stdout.decode()[:-1]
    return hostname


def get_gpu_name():
    ret = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                         stdout=subprocess.PIPE)

    # This gives us a lot of extra fluffy name information we don't need.
    name_full = ret.stdout.decode()

    # Do our best to parse the output GPU name.
    name = name_full.replace("NVIDIA", "").replace("GeForce", "").replace("RTX", "")
    return name.strip()


def check_make_results_root_dir():
    if not RESULTS_DIR.exists():
        print("Detected no previous results stored in the results directory.")
        print("Creating results directory:", RESULTS_DIR, end=" --- ")
        RESULTS_DIR.mkdir(parents=True)
        print("Success.")
        warn("Warning. Since you have no previous benchmark results, there's nothing to compare "
             "new results to.")


def make_rig_name(hostname: str, gpu_name: str):
    gpu_name_concise = gpu_name.replace(" ", "_")
    rig_name = f"{hostname}_{gpu_name_concise}"
    return rig_name


def main(python_executable: str):
    hostname = get_hostname()
    gpu_name = get_gpu_name()
    rig_name = make_rig_name(hostname, gpu_name)

    print("Detected the following system information:")
    print("\tHOSTNAME:", hostname)
    print("\tGPU Name:", gpu_name)

    check_make_results_root_dir()

    rig_results_dir = RESULTS_DIR / rig_name
    rig_results_dir.mkdir(exist_ok=True)

    benchmark_jobs = [
        BenchmarkJobConfig("arm_segmentation", Path("src/arm_segmentation/perf.py")),
        BenchmarkJobConfig("pytorch_kinematics", Path("tests/gen_fk_perf.py"))
    ]

    for job in benchmark_jobs:
        run_job(job, python_executable, rig_results_dir)


if __name__ == "__main__":
    python_executable = sys.executable
    main(python_executable)
