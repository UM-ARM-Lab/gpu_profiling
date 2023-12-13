from pathlib import Path
import re

REPO_DIR = Path(__file__).parent

PIPFILE_TEMPLATE = '''[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[[source]]
url = "https://download.pytorch.org/whl/cu{cuda_version_str}"
verify_ssl = true
name = "downloadpytorch"

[packages]
torch = {{version = "*", index = "downloadpytorch"}}
torchvision = {{version = "*", index = "downloadpytorch"}}
matplotlib = "*"
numpy = "*"
arm-segmentation = {{file = "arm_segmentation"}}
pytorch-kinematics = {{file = "pytorch_kinematics"}}

[dev-packages]

[requires]
python_version = "3"'''


def main():
    msg = ("Please input desired CUDA version for PyTorch install in the format of\n"
           "\t'<MAJOR_VERSION>.<MINOR_VERSION>' (no quotes) and hit Enter: ")
    cuda_version_str = input(msg)

    # Clean up user input by removing all non-numeric characters.
    cuda_version_str = re.sub("[^0-9]", "", cuda_version_str)

    if len(cuda_version_str) != 3:
        raise RuntimeError("Expected CUDA version string to be 3 numbers but got:",
                           cuda_version_str)

    pipfile_str = PIPFILE_TEMPLATE.format(cuda_version_str=cuda_version_str)
    pipfile_path = REPO_DIR / "Pipfile"
    with pipfile_path.open('w') as f:
        f.write(pipfile_str)
    print("Wrote Pipfile to:", pipfile_path)


if __name__ == "__main__":
    main()
