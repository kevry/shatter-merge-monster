from caveclient import CAVEclient

""" Use this script to find the latest root ids for POTENTIAL merge monsters """

""" 
Identifying MONSTER merge errors relies on the L2 cache https://caveclient.readthedocs.io/en/latest/guide/l2cache.html
- Segments that have more than a specified number of l2 nodes, are considered MONSTER merge errors 
- Chunked graph doc: https://caveclient.readthedocs.io/en/latest/guide/chunkedgraph.html
    
"""

################# Parameters (CHANGE OF NEEDED) #######################
client = CAVEclient(datastack_name='brain_and_nerve_cord') # Initilaize caveclient
MONSTER_THRESH = 500 # Threshold for number of l2 nodes to consider segment a merge monster
####################################################

def find_monster(root_id, client, MONSTER_THRESH):
    """ find merge monster using an initial root id """
    found_monster = False
    # original_root_id = root_id

    while not found_monster:
        potential_latest_root_id = client.chunkedgraph.get_latest_roots(root_id)


        if len(potential_latest_root_id) == 1: # only 1 parent root id found

            potential_root_id = potential_latest_root_id[0]
            num_l2_nodes = len(client.chunkedgraph.get_leaves(potential_root_id, stop_layer=2)) # get number of l2 nodes

            if potential_root_id == root_id: # if it matches, current root id, you found the latest
                print(f"- {root_id} is the latest merge monster with {num_l2_nodes} L2 nodes.")
            else:
                print(f"- {root_id} changed to {potential_root_id} with {num_l2_nodes} L2 nodes.")
                root_id = potential_latest_root_id[0]

            # found monster
            found_monster = True

        else:

            found_potential_monster = False
            # multiple potential root ids
            for potential_root_id in potential_latest_root_id:
                num_l2_nodes = len(client.chunkedgraph.get_leaves(potential_root_id, stop_layer=2))
                if num_l2_nodes > MONSTER_THRESH: # if one of parent ids is over the threshold, update root id
                    print(f"- {root_id} -> {potential_root_id}. New potential monster with {num_l2_nodes} nodes.")
                    root_id = potential_root_id
                    found_potential_monster = True
                    break
            if not found_potential_monster:
                print("- Error trying to find the latest merge monster segment!")
                return
    print('\n')
    return root_id


if __name__ == "__main__":

    # list of root ids that "might" be a merge monster
    #root_ids = [720575941412579183, 720575941486633734, 720575941514714691, 720575941515762019, 720575941519604440, 720575941521153715, 720575941557564068, 720575941590881790, 
    #            720575941595448433, 720575941611064138, 720575941615249049, 720575941645342369, 720575941658100248]
    
    root_ids = [720575941645373857, 720575941590881790]
    # loop through all to identify latest root id for these potential merge monsters
    for root_id in root_ids:
        print(f"Finding the new monster root id for {root_id}")
        _ = find_monster(root_id, client, MONSTER_THRESH=500)
