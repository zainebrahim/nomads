def gen_links(params):
    coll_img = 'collman'
    exp_img = 'M247514_Rorb_1_Site3Align2_EM'
    chan_img = 'em_clahe'

    coll_ann = 'collman'
    exp_ann = 'M247514_Rorb_1_Site3Align2_EM'
    chan_ann = 'xd'
    v_x = 2.25
    v_y = 2.25
    v_z = 2.25
    x = 1000
    y = 500
    z = 10
    image = "'{2}':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/{0}/{1}/{2}?'_'opacity':0.7}".format(coll_img, exp_img, chan_img)
    print(image)
    annotation = "'{2}':{type':'segmentation'_'source':'boss://https://api.boss.neurodata.io/{0}/{1}/{2}'_'opacity':0.7}".format(coll_ann, exp_ann, chan_ann)
    pos = "'navigation':{'pose':{'position':{'voxelSize':[{0}_{1}_{2}]_'voxelCoordinates':[{3}_{4}_{5}]}}_'zoomFactor':3}".format(v_x,v_y,v_z,x,y,z)
    link_glut = "https://viz.boss.neurodata.io/#!{'layers':{"+image+"_"+annotation+"}_"+pos+"}"


    #coll_ann = 'collman'
    #exp_ann = ' M247514_Rorb_1_Site3Align2_EM'
    #chan_ann =
    #v_x = 2.25
    #v_y = 2.25
    #v_z = 2.25
    #x = 1000
    #y = 500
    #z = 10
    #image = "'{2}':{'type':'image'_'source':'boss://https://api.boss.neurodata.io/{0}/{1}/{2}?'_'opacity':0.7}".format(coll_img, exp_img, chan_img)
    #annotation = "'{2}':{'type':'segmentation'_'source':'boss://https://api.boss.neurodata.io/{0}/{1}/{2}'_'opacity':0.7}".format(coll_ann, exp_ann, chan_ann)
    #pos = "'navigation':{'pose':{'position':{'voxelSize':[{0}_{1}_{2}]_'voxelCoordinates':[{3}_{4}_{5}]}}_'zoomFactor':3}".format(v_x,v_y,v_z,x,y,z)
    #link_gaba = "https://viz.boss.neurodata.io/#!{'layers':{"+image+"_"+annotation+"}_"+pos+"}"

    #return (link_glut, link_gaba)
    return link_glut

if(__name__ == '__main__'):
    print(gen_links('123'))
