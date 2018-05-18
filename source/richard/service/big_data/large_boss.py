import intern.utils.parallel as intern

class Block:
    def __init__(self, z_range, y_range, x_range):
        self.x_start = x_range[0]
        self.x_end = x_range[1]
        self.y_start = y_range[0]
        self.y_end = y_range[1]
        self.z_start = z_range[0]
        self.z_end = z_range[1]
        self.data = None

def get_data(host, token, col, exp, z_range, y_range, x_range):
    print("Downloading {} from {} with ranges: z: {} y: {} x: {}".format(exp,
                                                                         col,
                                                                         str(z_range),
                                                                         str(y_range),
                                                                         str(x_range)))
    resource = NeuroDataResource(host, token, col, exp)
    data_dict = {}
    blocks = intern.block_compute(x_start, x_end, y_start, y_end, z_start, z_end, (0, 0, 0), (5000, 5000, 20))
    for chan in resource.channels:
        merged_array = np.zeros(orig_shape)
        for block in blocks:
            x_r, y_r, z_r = blocks[i]
            merged_array[z_r[0]:z_r[1], y_r[0]:y_r[1], x_r[0]:x_r[1], :] = \
            resource.get_cutout(chan, z_r, y_r, x_r)
        data_dict[chan] = merged_array
        print(merged_array.shape)
    return data_dict


data = get_data("api.boss.neurodata.io", "56f34aad160c7bb424506690347b4d4b8814595f", "collman", "M247514_Rorb_1_light", [30, 35], [7000, 7500], [6000, 6500])
