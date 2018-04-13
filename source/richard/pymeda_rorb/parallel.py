import NeuroDataResource as ndr
import intern.utils.parallel as intern
import multiprocessing as mp
from util import Block, load_and_preproc
import sys
from importlib import import_module
import pickle
from functools import partial
import numpy as np
from skimage import measure
from skimage import filters
import pymeda_driver as driver


"""
    This function is designed to compute proper block sizes (less than 2 gb)
    when given a NDR
"""

def compute_blocks(resource):
    z, y, x = resource.max_dimensions
    x_start, x_end = resource.x_range
    y_start, y_end = resource.y_range
    z_start, z_end = resource.z_range

    block_size = (1000, 1000, 25)
    blocks = intern.block_compute(x_start, x_end, y_start, y_end, z_start, z_end, (0, 0, 0), block_size)
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
    raw_data = {}
    for key in resource.requested_channels:
        if key in resource.channels:
            raw = resource.get_cutout(chan = key, zRange = z_range, yRange=y_range, xRange=x_range)
            cutouts[key] = raw
            raw_data[key] = raw

    for key in resource.channels:
        if key not in resource.requested_channels:
            raw = resource.get_cutout(chan = key, zRange = z_range, yRange = y_range, xRange = x_range)
            raw_data[key] = raw
    block.request_data = cutouts
    block.raw = raw_data
    return block

def job(block, resource, function = None):

    print("Starting job, retrieiving data")
    block = get_data(resource, block)
    print("Starting algorithm")
    try:
        result = function(block.request_data)
    except Exception as ex:
        print(ex)
        print("Ran into error in algorithm, exiting this block")
        return

    synapse_labels = measure.label(result, background=0)
    connected_components = {}
    for z in range(synapse_labels.shape[0]):
        for y in range(synapse_labels.shape[1]):
            for x in range(synapse_labels.shape[2]):
                if (synapse_labels[z][y][x] in connected_components.keys()):
                    connected_components[synapse_labels[z][y][x]].append((z, y, x))
                else:
                    connected_components[synapse_labels[z][y][x]] = [(z, y, x)]
    connected_components.pop(0)

    synapse_centroids = []
    for key, value in connected_components.items():
        z, y, x = zip(*value)
        synapse_centroids.append((int(sum(z)/len(z)), int(sum(y)/len(y)), int(sum(x)/len(x))))

    data = block.raw
    data = load_and_preproc(data)
    data_dict = driver.get_aggregate_sum(synapse_centroids, data)
    df = driver.get_data_frame(data_dict)

    key = str(block.z_start) + "_" + str(block.y_start) + "_" + str(block.x_start)
    df.to_csv(key + ".csv", sep='\t')
    print("Done with job")
    return key

def run_parallel(config_file, cpus = None, function = None):
    ## Make resource and compute blocks
    resource = ndr.get_boss_resource(config_file)
    blocks = compute_blocks(resource)
    ## prepare job by fixing NeuroDataRresource argument
    task = partial(job, resource = resource, function = function)
    print("starting parallel")
    ## Prepare pool
    num_workers = cpus
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    pool = mp.Pool(num_workers)
    try:
        print(pool.map(task, blocks))
    except:
        pool.terminate()
        print("Parallel failed, closing pool and exiting")
        raise
    pool.terminate()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Provide module, function as arguments")
        sys.exit(-1)
    #TODO: integrate argparser
    mod = import_module(sys.argv[1])
    function = getattr(mod, sys.argv[2])
    run_parallel("neurodata.cfg", function = function)
