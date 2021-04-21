"""
Generate a torchbench test report from a file containing the PR body.
Currently, only supports running tests on specified model names

Testing environment:
- Intel Xeon 8259CL @ 2.50 GHz, 24 Cores with disabled Turbo and HT
- Nvidia Tesla T4
- Nvidia Driver 450.51.06
- Python 3.7
- CUDA 10.2
"""
# Known issues:
# 1. Does not reuse the build artifact in other CI workflows
# 2. CI jobs are serialized because there is only one worker
import os
import git
import enum
import json
import argparse
import subprocess

from typing import Optional, List

CUDA_VERSION = "cu102"
PYTHON_VERSION = "3.7"
THRESHOLD = 5
PYTORCH_PKGS: List[str] = ["pytorch", "text", "vision", "benchmark"]
PYTORCH_BUILD_CMDS = {
    "pytorch": "python setup.py install",
    "text": "python setup.py clean install",
    "vision": "python setup.py install",
    "benchmark": "python install.py",
}
PR_SRC_ROOT: str = os.path.join(os.environ("HOME"), ".torchbench", "pr", "src")
OUTPUT_DIR: str = os.path.join(os.environ("HOME"), ".torchbench", "pr", os.environ("GITHUB_RUN_ID"))

class GitRepo:
    name: str
    repo_path: str
    build_cmd: str
    def __init__(self, name: str, repo_path: str):
        self.name = name
        self.repo_path = repo_path
        self.build_cmd = PYTORCH_BUILD_CMDS[name]
    def cleanup(self):
        pass
    def checkout(self, sha: str):
        pass
    def sync(self):
        pass
    def build(self):
        pass
    def check_sha_exist(self, sha: str) -> bool:
        pass

# TODO: run TorchBench with the filter
def run_torchbench(repo_path: str, filter: str, out_dir: str):
    pass

def uninstall_pip_packages(pkgs: List[str]):
    # No need to uninstall the "benchmark" package
    pkgs = filter(lambda p: not p == "benchmark", pkgs)
    command = ["pip", "uninstall", "-y"]
    command.extend(pkgs)
    subprocess.check_call(command)

# os.symlink doesn't have force option, so run with subprocess
def force_soft_link(src: str, dst: str):
    command = ["ln", "-sf", src, dst]
    subprocess.check_call(command)

# Validate that the current torch version matches the given sha
def validate_build(sha: str) -> bool:
    command = ["python", "-c", "import torch; print(torch.version.git_version)"]
    output = subprocess.check_output(command, shell=True).decode().strip()
    return output[:len(sha)] == sha

def get_latest_json_from_dir(d: str) -> str:
    for fname in reverse(sorted(os.listdir(d))):
        if fname.endswith(".json"):
            return fname
    print(f"Do not find json file in dir: {d}")
    exit(1)

def get_result_means(jdata):
    rc = { }
    for param in jdata["benchmarks"]:
        name = param["name"]
        mean = param["stats"]["mean"]
        rc[name] = mean
    return rc
    
def gen_pr_report(control_dir: str, treatment_dir: str, out_dir: str):
    control_json = get_latest_json_from_dir(control_dir)
    treatment_json = get_latest_json_from_dir(treatment_dir)
    # Link jsons to out_dir
    force_soft_link(control_json, os.path.join(out_dir, "control.json"))
    force_soft_link(treatment_json, os.path.join(out_dir, "treatment.json"))
    # Generate overview.txt
    out = [['Benchmark']]
    with open(control_json, "r") as fp:
        control_job = json.load(fp)
        control_means = get_means(control_job)
    for key in control_means:
        out.append([])
        out[-1].append(key)
    for index, json_file in [control_json, treatment_json]:
        with open(json_file, "r") as fp:
            obj = json.load(fp)
        header = f'Run {obj["pytorch"]["git_version"]}'
        out[0].append(header)
        means = get_means(obj)
        if index == 0:
            reference = means
        for key_index, key in enumerate(means):
            if index == 0 or index == 1:
                out[key_index+1].append(means[key])
            if index == 1:
                # Append deltas
                delta_num = ((means[key] - reference[key]) / means[key] * 100)
                delta_str = "{:+3f}".format(delta_num) + "%" 
                if (abs(delta_num) >= THRESHOLD):
                    delta_str = delta_str + "*"
                out[key_index+1].append(delta_str)
    out_str = tabulate(out, headers='firstrow')
    with open(os.path.join(out_dir, "overview.txt"), 'w') as ofp:
        oft.write(out_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TorchBench tests based on PR')
    parser.add_argument('pr-num', required=True, type=str, help="The Pull Request number")
    parser.add_argument('pr-base-sha', required=True, type=str, help="The Pull Request base hash")
    parser.add_argument('pr-tip-sha', required=True, type=str, help="The Pull Request tip hash")
    parser.add_argument('pr-body', required=True, type=argparse.FileType('r'),
                        help="The file that contains body of a Pull Request")
    parser.add_argument('torchbench-path', required=True, type=str, help="Path to TorchBench repository")
    args = parser.parse_args()
    
    # Preparation: uninstall packages, clean the source directory and sync with origin
    uninstall_pip_packages(PYTORCH_PKGS)
    repos: Dict[str, GitRepo] = { }
    for pkg in PYTORCH_PKGS:
        repos[pkg] = GitRepo(name = pkg, repo_path = os.path.join(PR_SRC_ROOT, pkg))
        repos[pkg].cleanup()
        repos[pkg].checkout("master")
        repos[pkg].sync()
    # Sanity check: make sure the two shas exist in the git repo
    repos["pytorch"].check_sha_exist(args.pr_base_sha)
    repos["pytorch"].check_sha_exist(args.pr_tip_sha)
    # Identify the specified models, verify the input
    bench_filter = extract_models_from_pr(args.torchbench_path, args.pr_body)
    if not bench_filter:
        print(f"Can't parse the model filter from the pr body. Currently we only support allow-list.")
        return
    print(f"Ready to run TorchBench with benchmark filter {torchbench_filter}. Result will be saved in the directory: {OUTPUT_DIR}.")
    # Build the base packages
    for pkg in PYTORCH_PKGS:
        repo = repos[pkg]
        if pkg == "pytorch":
            repo.checkout(args.pr_base_sha)
        repo.build()
    validate_build()
    # Run TorchBench with the filter
    control_out_dir = os.path.join(OUTPUT_DIR, args.pr_base_sha)
    run_torchbench(repos["benchmark"], bench_filter, control_out_dir)
    # Uninstall the base pytorch and other dependencies
    uninstall_pip_packages(PYTORCH_PKGS)
    # Build the PR tip packages
    for pkg in PYTORCH_PKGS:
        repo = repos[pkg]
        if pkg == "pytorch":
            repo.checkout(args.pr_tip_sha)
        repo.build()
    validate_build()
    # Run TorchBench with the filter
    treatment_out_dir = os.path.join(OUTPUT_DIR, args.pr_tip_sha)
    run_torchbench(repos["benchmark"], bench_filter, treatment_out_dir)
    # Uninstall the PR tip pytorch and other dependencies
    uninstall_pip_packages(PYTORCH_PKGS)
    # Output the PR performance impact report
    # A report contains three files: control.json, treatment.json, and overview.txt
    gen_pr_report(control_out_dir, treatment_out_dir, os.path.join(OUTPUT_DIR, "overview.txt"))
