import copy
from functools import partial
from itertools import chain, cycle

import numpy as np


def generate_batches(batch_size, products, totals, sample_size):
    sizes = (
        (np.ceil(np.array(totals) / np.min(totals)) * batch_size).astype(int).tolist()
    )
    iterators = [[product] * size for product, size in zip(products, sizes)]
    base = cycle(chain(*iterators))
    return list(dict(zip(range(sample_size), base)).values())


def no_priority(machine_tuple, product=None):

    if product is None:
        raise RuntimeError("No product set")

    return machine_tuple[0]


def sum_of_product_priority(machine_tuple, product=None):

    if product is None:
        raise RuntimeError("No product set")

    return sum(machine_tuple[1].values())


def product_count_priority(machine_tuple, product=None):

    if product is None:
        raise RuntimeError("No product set")

    return machine_tuple[1][product]


def double_priority(machine_tuple, product=None):

    if product is None:
        raise RuntimeError("No product set")

    return (machine_tuple[1][product], sum(machine_tuple[1].values()))


def simulate(
    num_of_machines,
    sequence,
    products,
    totals,
    flush_lock,
    transition_lock,
    priority,
    reverse=True,
):

    mix = dict(zip(products, totals))
    machines = {i: {p: 0 for p in products + ["lock"]} for i in range(num_of_machines)}
    flushes = {i: 0 for i in range(num_of_machines)}
    trash = []

    data = []
    trash_data = []
    consumption_data = []

    for product in sequence:

        consumed = False
        for i, mach in sorted(
            machines.items(), key=partial(priority, product=product), reverse=reverse
        ):

            # Machine is available to accept products
            machine_is_available = machines[i]["lock"] == 0

            # Machine has space to accept product
            product_fits = machines[i][product] < mix[product]

            # Complete cases for 3 binary variables
            if not consumed and machine_is_available and product_fits:
                machines[i][product] += 1
                machines[i]["lock"] = transition_lock
                consumed = True

            elif not consumed and machine_is_available and not product_fits:
                continue

            elif not consumed and not machine_is_available and product_fits:
                machines[i]["lock"] -= 1

            elif not consumed and not machine_is_available and not product_fits:
                machines[i]["lock"] -= 1

            elif consumed and machine_is_available and product_fits:
                continue

            elif consumed and machine_is_available and not product_fits:
                continue

            elif consumed and not machine_is_available and product_fits:
                machines[i]["lock"] -= 1

            elif consumed and not machine_is_available and not product_fits:
                machines[i]["lock"] -= 1

        # Trash un-consumed products
        if not consumed:
            trash.append(product)

        # Save the data
        data.append(copy.deepcopy(machines))
        trash_data.append(len(trash))

        # Flush ready machines
        amount_consumed = 0
        for j in machines:
            if all(machines[j][p] == mix[p] for p in products):
                machines[j] = {p: 0 for p in products + ["lock"]}
                machines[j]["lock"] = flush_lock
                flushes[j] += 1
                amount_consumed += sum(mix.values())

        consumption_data.append(amount_consumed)

    return {
        "summary": {
            "trash": len(trash),
            "total": len(sequence),
            "consumed": len(sequence) - len(trash),
            "flushes": flushes,
        },
        "data": data,
        "trash_data": trash_data,
        "consumption_data": np.cumsum(consumption_data),
    }
