# Programatically shatter merge monsters

## How to run

### Config file
Update the config.yaml to the root id you want to programatically split. Modify any parameters if needed.

`cd shatter-merge-monster`

### Install libs
`pip install -r requirements.txt`


### Run script
To start shattering, run the following:

`python3 monster_shatter.py`

Data will be saved to
- `logs/{specific_dir}`: holds the log files generated when running the script.
- `executed_splits/{specific_dir}`: holds the json files created after every split.


### Helper tools
Use `find_monster.py` to find the LATEST root ids for a given set of root ids. 