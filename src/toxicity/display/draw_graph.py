import matplotlib.pyplot as plt
from typing import Any
import logging
from taskman_client.named_number_proxy import NamedNumberProxy
import matplotlib.pyplot as plt


def draw_graph(output):

    # Assuming output is a list of tuples (step, val)
    steps, values = zip(*output)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, marker='o')

    # plt.ylim(89.5, 95.5)  # Slightly expanded range for better visibility

    plt.title('toxigen_test Scores Over Steps')
    plt.xlabel('Step')
    plt.ylabel('toxigen_test Score')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate each point with its exact value
    for i, (step, val) in enumerate(output):
        plt.annotate(f'{val:.2f}', (step, val), textcoords="offset points", xytext=(0, 10), ha='center')

    # Use log scale for x-axis due to large step differences
    plt.xscale('symlog')  # symlog allows zero values
    plt.xticks(steps, [str(step) for step in steps])  # Ensure all step labels are shown

    plt.tight_layout()
    plt.show()

def main():
    log = logging.getLogger(__name__)
    search = NamedNumberProxy()
    target_field = "auc"

    def load_method_relate_scores(method):
        ret = search.search_numbers(method)
        print(ret)
        d: dict[str, Any] = {}
        for e in ret:
            if e['field'] == target_field and e['name'] == method:
                key = e['condition']
                if key in d:
                    log.warning("key=%s is duplicated originally %s, replace with %s",
                                  str(key), str(d[key]), str(e['number']))
                try:
                    d[e['condition']] = e['number']
                except:
                    pass
        return d


    condition = "toxigen_test"
    output = []
    for step in [0, 1, 2, 5, 10, 50, 100, 200, 5000, 10000]:
        if step > 0:
            run_name= f"lg2_ft_{step}"
        else:
            run_name = "llama_guard2_prompt"

        d = load_method_relate_scores(run_name)
        val = d[condition]

        output.append((step, val))

    print(output)
    draw_graph(output)


if __name__ == "__main__":
    main()