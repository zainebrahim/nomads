import NeuroDataResource as ndr
import intern.utils.parallel as intern
import multiprocessing as mp
from util import Block
import pipeline as pipeline
import pickle
from functools import partial
from skimage.measure import block_reduce as pool
import numpy as np


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
    y_range = [block.y_start, block.y_end]
    x_range = [block.x_start, block.x_end]
    z_range = [block.z_start, block.z_end]
    cutouts = {}

    for key in resource.requested_channels:
        if key in resource.channels:
            raw = resouce.get)cutout(chan = key, zRange = z_range, yRange=y_range, XRange=x_range)
            cutouts[key] = raw




    block.data = cutouts
    return block

def job(block, resource):

    print("Starting job, retrieiving data")
    block = get_data(resource, block)

    try:
        result = pipeline.pipeline(block.data)
    except Exception as ex:
        print(ex)
        print("Ran into error in algorithm, exiting this block")
        return

    key = str(block.z_start) + "_" + str(block.y_start) + "_" + str(block.x_start)
    pickle.dump(result, open(key, "wb"))
    print("Done with job")
    return key

def run_parallel(config_file, cpus = None):
    ## Make resource and compute blocks
    resource = ndr.get_boss_resource(config_file)
    blocks = compute_blocks(resource)
    ## prepare job by fixing NeuroDataRresource argument
    job = partial(job, resource = resource)

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
    #resource = ndr.get_boss_resource("neurodata.cfg")
    #nomads(Block((50, 60), (5000, 6000), (5000, 6000)), resource)
