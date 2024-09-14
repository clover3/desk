import subprocess
import re

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        print(f"Error: {error.decode('utf-8')}")
    return output.decode('utf-8')

import re

def parse_gpu_info(text):
    print(text)
    pattern = r'gpu:(\w+):(\d+)(?:\(S:(\d+)-(\d+)\))?'
    match = re.match(pattern, text)

    if match:
        gpu_info = {
            'resource': 'gpu',
            'model': match.group(1),
            'total': int(match.group(2))
        }

        if match.group(3) and match.group(4):
            gpu_info['socket_start'] = int(match.group(3))
            gpu_info['socket_end'] = int(match.group(4))
        return gpu_info
    else:
        return None


def parse_sinfo(output):
    nodes = {}
    for line in output.strip().split('\n')[1:]:  # Skip header
        node, state, gpus = line.split()
        if gpus != "(null)":
            info = parse_gpu_info(gpus)
            info["state"] = state
            nodes[node] = info
    return nodes



def parse_squeue(output):
    gpu_usage = {}
    for line in output.strip().split('\n')[1:]:  # Skip header
        print(line)
        parts = line.split()
        if len(parts) >= 8:
            node = parts[6]
            if 'gpu' in parts[7]:
                tokens = parts[7].split(':')
                try:
                    n_used = int(tokens[-1])
                except ValueError:
                    if parts[7].startswith('gres/gpu'):
                        n_used = 1
                    else:
                        print("cannot parse ", parts[7])
                        raise

                if node not in gpu_usage:
                    gpu_usage[node] = 0
                gpu_usage[node] += n_used
    return gpu_usage


def main():
    sinfo_output = run_command("sinfo -o \"%n %T %G\"")
    squeue_output = run_command("squeue -t R -o \"%.18i %.9P %.8j %.8u %.10M %.6D %R %b\"")

    nodes = parse_sinfo(sinfo_output)
    gpu_usage = parse_squeue(squeue_output)

    for node, usage in gpu_usage.items():
        if node in nodes:
            nodes[node]['used'] = usage

    print("GPU Usage Summary:")
    print("------------------")
    total_gpus = 0
    total_used = 0
    for node, info in nodes.items():
        total = info['total']
        try:
            used = info['used']
        except KeyError:
            used = 0
        print(f"{node}\t{info['model']}\t{info['state']}\t{used}/{total} GPUs used")
        total_gpus += total
        total_used += used

    print("\nOverall Summary:")
    print(f"Total GPUs: {total_gpus}")
    print(f"GPUs in use: {total_used}")
    print(f"Utilization: {total_used/total_gpus*100:.2f}%")

if __name__ == "__main__":
    main()