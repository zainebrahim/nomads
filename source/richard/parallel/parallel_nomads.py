import NeuroDataResource as ndr
import intern.utils.parallel as intern
import multiprocessing as mp
from util import Block
import NOMADS_pipeline as algo
import pickle
from functools import partial

"""
    This function is designed to compute proper block sizes (less than 2 gb)
    when given a NDR
"""

def compute_blocks(resource):
    z, y, x = resource.max_dimensions
    block_size = (1000, 1000, 10)
    blocks = intern.block_compute(0, x, 0, y, 0, z, (0, 0, 0), block_size)
    ### IMPORTANT blocks are returned as x, y, z ###
    for i in range(len(blocks)):
        x_range, y_range, z_range = blocks[i]
        blocks[i] = Block(z_range, y_range, x_range)
    return blocks

def get_data(resource, block):
    data = {}
    z_range = [block.z_start, block.z_end]
    y_range = [block.y_start, block.y_end]
    x_range = [block.x_start, block.x_end]
    for chan in resource.channels:
        if "DAPI" in chan:
            continue
        else:
            data[chan] = resource.get_cutout(chan, z_range, y_range, x_range)
    block.data = data
    print(data["GABA"])
    return block

def nomads(block, resource):
    block = get_data(resource, block)
    result = algo.pipeline(block.data)
    key = str(block.z_start) + "_" + str(block.y_start) + "_" + str(block.x_start)
    pickle.dump(open(key, "wb"), result)
    return key

def run_parallel(config_file, cpus = None):
    ## Make resource and compute blocks
    resource = ndr.get_boss_resource(config_file)
    blocks = compute_blocks(resource)
    ## prepare job by fixing NeuroDataRresource argument
    job = partial(nomads, resource = resource)

    ## Prepare pool
    num_workers = cpus
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    pool = mp.Pool(num_workers)
    try:
        print(pool.map(job, blocks))
    except:
        pool.terminate()
        print("Parallel failed, closing pool and exiting")
        raise
    pool.terminate()

if __name__ == "__main__":
    run_parallel("neurodata.cfg")
