import NeuroDataResource as ndr
import intern.utils.parallel as intern
import multiprocessing as mp
from util import Block
import NOMADS_pipeline as algo
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
    block_size = (5000, 5000, 10)
    blocks = intern.block_compute(0, x, 0, y, 0, z, (0, 0, 0), block_size)
    ### IMPORTANT blocks are returned as x, y, z ###
    for i in range(len(blocks)):
        x_range, y_range, z_range = blocks[i]
        blocks[i] = Block(z_range, y_range, x_range)
    return blocks

def get_data(resource, block):
    y_range = [block.y_start, block.y_end]
    x_range = [block.x_start, block.x_end]

    cutouts = {}

    for i in range(block.z_start, block.z_end):
        for key in resource.channels:
            if "psd" in key.lower() or "synapsin" in key.lower():
                raw = resource.get_cutout(chan=key, zRange=[i, i+1], yRange=y_range, xRange=x_range)[0]
                if (raw.dtype != np.dtype("uint8")):
                    info = np.iinfo(raw.dtype) # Get the information of the incoming image type
                    raw = raw.astype(np.float64) / info.max # normalize the data to 0 - 1
                    raw = 255 * raw # Now scale by 255
                    raw = raw.astype(np.uint8)
                cutout = pool(raw, (36, 36), np.mean)
                if key in cutouts.keys():
                    cutouts[key].append(cutout)
                else:
                    cutouts[key] = [cutout]
    data = []
    for elem in cutouts:
        data.append(np.stack(cutouts[elem]))
    data = np.stack(data)
    print(data.shape)
    #if len([True for elem in data.shape[1:] if elem < 20]):
    #    print("Skipping block")
    #    return

    block.data = data
    return block

def nomads(block, resource):
    print("starting job")
    block = get_data(resource, block)
    try:
        result = algo.pipeline(block.data)
    except Exception as ex:
        print(ex)
        print("ran into error in algo, exiting this chunk")
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
    #resource = ndr.get_boss_resource("neurodata.cfg")
    #nomads(Block((50, 60), (5000, 6000), (5000, 6000)), resource)
