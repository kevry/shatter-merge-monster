# Programatically shatter merge monsters in the BANC Dataset
"Shatter" heavily merged segmented areas in the BANC dataset

## Description of process (05/9/2025)
The `monster_shatter.py` script uses the l2 cache to skeletonize the segment. Converts that to a graph and uses the louvian method to create communities. Then it tries to pull each community apart from the original segment one by one.

This uses CAVEclient so you must have an authentication token already setup. 

Read directions to setup an auth token :
https://caveclient.readthedocs.io/en/latest/guide/authentication.html

## How to run

### Clone repository
` git clone https://github.com/kevry/shatter-merge-monster.git`

`cd shatter-merge-monster`


### Modify config file (if needed)
Update the `config.yaml`.  Modify the ROOT_ID_2_USE paramater to the merge monsters root id you want to shatter. Modify any parameters if needed. Update the SUB_DIRECTORY fto create a sub directory for tracking.

### Install libraries
`pip install -r requirements.txt`

### Run script
To start shattering, run the following (the script will use the config file):

`python3 monster_shatter.py`

Data will be saved to
- `logs/{specific_dir}`: holds the log files generated when running the script.
- `executed_splits/{specific_dir}`: holds the json files created after every split.


### Helper tools
Use `find_monster.py` to find the LATEST root ids for a given set of root ids. (Root ids from the proofreadable segmentation layer often change)


##