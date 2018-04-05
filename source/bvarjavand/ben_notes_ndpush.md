Ingest job class (creates a class for the ingest job, iterates in chunks of z, both pulling and posting in blocks)
Ingest large vol processes arguments and creates an ingest job
for loop pulling 16 slices at a time
Inside ingest_job, takes all CLA, and processes if needed

In ingest_job there are if statements for doing different things

ingest_job needs to create a new class (polygons) that will be created when ingest_job is initialized

Command line flag, but the way its structured is that ingest_large_vol passes things through
ingest_job takes the arguments and figures out what class it is

test_render_resource

ingest_job.py line 78 but with polygonResource

read_img_stack with ingest_job

Create CLA for polygons in ingest_large_vol

gen_commands: do this for polygons
#
only used for 'render' source_type

render_owner = 'OWNER_NAME'

render_project = 'PROJECT_NAME'

render_stack = 'STACK_NAME'

render_channel = 'CHANNEL_NAME'  # can be None if no channels in the stack

render_baseURL = 'BASEURL'
#


If the data source is not polygon source,


pytest.raises assertion error
