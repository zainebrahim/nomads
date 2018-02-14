limited amount of memory - tif being loaded in isnt trivial

ndwebtools fails beyond a certain size limit (cutout above some thing times out the request).

you will run into right locks because the boss has bugs - the boss stores as cubes and will be super slow (so we want to do it all at once)

do all your writes or reads at once, and then optimize how to do writes at once (get the indices right in the for loop).

download data to annotate, make annotations, upload it.

tiffile can attach metadata to tif files easily (key:value pairs (version number of form, annotation identifier))

validation in the tif you pulled that checks you're in the same spot (x,y,z, channel, exp, coll, datatype, ann-id, v# of form)

buncha pull requests - run it locally, port 8000 or 8080 (keycloak will work).
