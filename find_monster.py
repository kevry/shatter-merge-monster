from caveclient import CAVEclient

""" Use this script to find the latest root ids for POTENTIAL merge monsters """

# initilaize caveclient
client = CAVEclient(datastack_name='brain_and_nerve_cord')

def find_monster(client, root_id, monster_thresh=5000):
    """ find merge monster using an initial root id """
    found_monster = False
    original_root_id = root_id
    
    while not found_monster:
        potential_latest_root_id = client.chunkedgraph.get_latest_roots(root_id)
        if len(potential_latest_root_id) == 1: 
            if potential_latest_root_id[0] == root_id:
                n_l2_nodes = len(client.chunkedgraph.get_leaves(root_id, stop_layer=2))
                print(f"- {root_id} is the latest merge monster with {n_l2_nodes} nodes.")
            else:
                n_l2_nodes = len(client.chunkedgraph.get_leaves(root_id, stop_layer=2))
                print(f"- {root_id} changed to {potential_latest_root_id[0]} with {n_l2_nodes} nodes.")
                root_id = potential_latest_root_id[0]
            found_monster = True
        else:
            for pid in potential_latest_root_id:
                n_l2_nodes = len(client.chunkedgraph.get_leaves(pid, stop_layer=2))
                if n_l2_nodes > monster_thresh:
                    print(f"- {root_id} -> {pid}. New potential monster with {n_l2_nodes} nodes.")
                    root_id = pid
                    break
    print('\n')
    return root_id


if __name__ == "__main__":
    # root_id = 720575941352166960

    # list of root ids to test
    root_ids = [720575941352166960, 720575941403906416, 720575941525844580, 720575941534703809, 
                720575941573984777, 720575941593317669, 720575941595424133, 720575941617092053, 720575941654695892]
    
    for root_id in root_ids:
        print(f"Finding the new monster root id for {root_id}")
        find_monster(client, root_id, monster_thresh=2000)