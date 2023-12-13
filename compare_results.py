from typing import Dict, List, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class SegmentationResults:

    def __init__(self, results_dir: Path):
        self.results_file: Path = results_dir / "gpu_profiling_arm_segmentation.txt"
        self.train, self.inference = self._read_segmentation_results()

    def _read_segmentation_results(self):
        with self.results_file.open('r') as f:
            res_lines = f.readlines()

        if not res_lines[1].startswith("Train"):
            raise RuntimeError("Expected line 1 (the second line) to start with 'Train:'")
        res_train_secs = float(res_lines[1].split(':')[1])

        if not res_lines[2].startswith("Inference"):
            raise RuntimeError("Expected line 2 (the third line) to start with 'Inference:'")
        res_inference_secs = float(res_lines[2].split(':')[1])

        return (res_train_secs, res_inference_secs)


class KinematicsResults:

    def __init__(self, results_dir: Path):
        self.results_file: Path = results_dir / "gpu_profiling_pytorch_kinematics.txt"

        # Results organized as dictionary with job name->device->dtype->batch_size->time (seconds)
        self.results = self._read_and_parse_results()

    def _read_and_parse_results(self):
        with self.results_file.open('r') as f:
            res_lines = f.readlines()

        results = dict()
        for line in res_lines:
            if len(line) <= 2:
                break

            name, device, dtype, batch_size, t_sec = self._split_line(line)

            d_name = self._ensure_dict(results, name)
            d_device = self._ensure_dict(d_name, device)
            d_dtype = self._ensure_dict(d_device, dtype)

            d_dtype[batch_size] = t_sec

        return results

    def _ensure_dict(self, root_dict, key):
        if key not in root_dict:
            root_dict[key] = dict()
        return root_dict[key]

    def _split_line(self, line: str):
        name, device, dtype, batch_size, t_sec = line.split()
        name = name.split('=')[1].replace("'", '')
        device = device.split('=')[1].replace("'", '')
        dtype = dtype.split('=')[1]
        batch_size = int(batch_size.split('=')[1])
        t_sec = float(t_sec)
        return name, device, dtype, batch_size, t_sec


class GpuProfilingResults:

    def __init__(self, results_dir: Path):
        self.results_dir: Path = results_dir
        self.rig_name, self.gpu_name = self._parse_dir_name()
        self.segmentation = SegmentationResults(results_dir)
        self.kinematics = KinematicsResults(results_dir)

    def get_clean_name(self):
        return f"{self.rig_name} {self.gpu_name}"

    def _parse_dir_name(self):
        """Parses the directory name into Rig and GPU names"""
        dir_strs = self.results_dir.stem.split('_')
        if len(dir_strs) < 2:
            raise RuntimeError("Results directory for a rig should be named <HOSTNAME>_<GPU_TYPE>")

        rig_name = dir_strs.pop(0).title()
        gpu_name = " ".join(dir_strs)
        return rig_name, gpu_name


class GpuProfilingResultsComparison:

    def __init__(self, results_root_dir: Path, output_dir: Optional[Path] = None):
        self.results_root_dir: Path = results_root_dir
        self.output_dir: Path = self._handle_output_dir(output_dir)
        self.profiles: Dict[str, GpuProfilingResults] = self._read_results()

    def plot_segmentation_results(self, rig_order: List[str] = None):
        """Plots results for arm-segmentation GPU profiling"""
        print("Plotting arm-segmentation results")
        fig, ax = plt.subplots()

        if rig_order is None:
            rig_order = self.profiles.keys()

        attrs = ["Train", "Inference"]
        num_attrs = len(attrs)
        x = np.arange(num_attrs)  # the label locations
        width = (1. / (num_attrs + 2))  # the width of the bars
        multiplier = 0

        for rig_name in rig_order:
            results = results_grouped.profiles[rig_name]
            measurements = [results.segmentation.train, results.segmentation.inference]
            offset = width * multiplier

            rig_name_clean = results.get_clean_name()
            rects = ax.bar(x + offset, measurements, width, label=rig_name_clean)

            bar_labels = [f"{m:.2f}" for m in measurements]
            ax.bar_label(rects, padding=3, labels=bar_labels)
            multiplier += 1

        ax.set_ylabel('Time (seconds)')
        ax.set_xticks(x + width, ["Train", "Inference"])
        ax.legend()

        self._adjust_ylim(ax, 1.2)

        fig.suptitle("arm-segmentation GPU Profiling")
        fig.tight_layout()

        figpath = self.output_dir / "armsegmentation_comparison.png"
        fig.savefig(figpath, dpi=300)

        print("\tSaved plot to:", figpath)

        return fig

    def plot_pk_results(self,
                        rig_order: Optional[List[str]] = None,
                        convert_to_ms: bool = True,
                        job_names: List[str] = ["val", "val_serial", "kuka_iiwa"]):
        """Plots results for pytorch-kinematics GPU profiling"""
        print("Plotting pytorch-kinematics results")
        if rig_order is None:
            rig_order = list(self.profiles.keys())

        figs = []
        for job in job_names:
            fig = self._plot_pk_job_result(job, rig_order, convert_to_ms=convert_to_ms)
            figpath = self.output_dir / f"pk_{job}_comparison.png"
            fig.savefig(figpath, dpi=300)
            print("\tSaved fig to:", figpath)
            figs.append(fig)
        return figs

    def _handle_output_dir(self, output_dir: Path):
        if output_dir is None:
            output_dir = self.results_root_dir / ".." / "comparison_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _read_results(self):
        profiles: Dict[str, GpuProfilingResults] = dict()
        for child in self.results_root_dir.iterdir():
            if not child.is_dir():
                continue
            print("Reading results for:", child)
            profiles[child.stem] = GpuProfilingResults(child)
        return profiles

    def _plot_pk_job_result(self,
                            job_name: str,
                            rig_order: List[str],
                            device: str = "cuda",
                            dtypes: List[str] = ["torch.float32", "torch.float64"],
                            batch_sizes: List[int] = [1e3, 1e4, 1e5],
                            convert_to_ms: bool = True):
        num_attrs = len(batch_sizes)
        x = np.arange(num_attrs)
        width = (1. / (num_attrs + 2))  # the width of the bars

        fig, ax_array = plt.subplots(nrows=2, sharex=True, figsize=[6.4, 6.4])
        for i, dtype in enumerate(dtypes):
            ax = ax_array[i]
            ax.set_title(dtype)

            for j, name in enumerate(rig_order):
                offset = j * width

                # Collect results for all jobs for this rig.
                res = self.profiles[name].kinematics.results
                ts = np.array([res[job_name][device][dtype][bs] for bs in batch_sizes])

                if convert_to_ms:
                    ts *= 1000.

                name_clean = self.profiles[name].get_clean_name()
                rects = ax.bar(x + offset, ts, width, label=name_clean)

                if convert_to_ms:
                    bar_labels = [f"{m:.0f}" for m in ts]
                else:
                    bar_labels = [f"{m:.2f}" for m in ts]

                ax.bar_label(rects, padding=3, label=bar_labels)

            if i == 0:
                ax.legend()

            y_units = "ms" if convert_to_ms else "seconds"
            ax.set_ylabel(f"Time ({y_units})")

            self._adjust_ylim(ax)

        ax.set_xticks(x + width, batch_sizes)
        ax.set_xlabel("Batch Size")
        fig.suptitle(f"pytorch-kinematics GPU Profiling - \"{job_name}\" Job")
        fig.tight_layout()

        return fig

    def _adjust_ylim(self, ax, buffer: float = 1.1):
        """Adjusts the Y axis limit to be slightly higher to avoid legend/bar/border overlap"""
        ymin, ymax = ax.get_ylim()
        ymax = buffer * (ymax - ymin) + ymin
        ax.set_ylim(ymin, ymax)


if __name__ == "__main__":
    results_grouped = GpuProfilingResultsComparison(Path("./results"))

    rig_order = ["armeclipse_1080_Ti", "armeclipse_4070", "legion_4090"]
    _ = results_grouped.plot_pk_results(rig_order)
    _ = results_grouped.plot_segmentation_results(rig_order)
