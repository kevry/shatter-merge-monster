from caveclient import CAVEclient
from datetime import datetime
import numpy as np
import os
import sys
import time
from utils import *
import yaml

# read configuration file (config.yaml)
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)


if config["SUB_DIRECTORY"]:
    config["SUB_DIRECTORY"] = str(config["SUB_DIRECTORY"])
    log_dir = f'logs/{config["SUB_DIRECTORY"]}'
    splits_dir = f'executed_splits/{config["SUB_DIRECTORY"]}'
else:
    log_dir = 'logs'
    splits_dir = 'executed_splits'

# create helper directories
os.makedirs(log_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

# direct standard output to log file for tracking
now = datetime.now().strftime("%Y%m%d%H%M%S")
log_file = open(os.path.join(log_dir, f'output_{now}.log'), 'w')
sys.stdout = OutBoth(sys.stdout, log_file)

# init caveclient
client = CAVEclient(datastack_name=config["CAVECLIENT_DATASTACK"])

# user input for root_id
# root_id_2_use = 720575941352166960 # MODIFY if using variable in script
if len(sys.argv) > 1:
    root_id = sys.argv[1]
else:
    root_id = config['ROOT_ID_2_USE']


if __name__ == "__main__":
    print("---- Shattering merge monsters ----")
    print(f'Starting off with root id: {root_id}\n')

    for ri in range(config["ROUNDS"]):
        # check if initial root id is the latest
        root_id, is_latest = check_latest_root_id(client, latest_root_id = root_id, verbose=True, replace_root_id=False)
        if not is_latest:
            raise ValueError(f"(ERROR) {root_id} is not the latest root id. Stopping script early.")

        is_monster = is_monster_segment(client, root_id, verbose=True, thresh_n_nodes = config["MONSTER_THRESH"])
        if not is_monster:
            raise ValueError(f"(ERROR) {root_id} is not a monster root id. Stopping script early.")
            
        l2_cache_success = False
        while not l2_cache_success: 
            # get the L2 cache data for the root id. If fails, try again every WAIT_SEC seconds
            l2stats, lvl2_eg, G, l2_cache_success = get_root_id_l2_data(client=client, root_id=root_id, return_graph=True)
            if not l2_cache_success:
                time.sleep(config["WAIT_SEC"]) # wait to retry
            
        # use l2 graph to cluster
        clusters = create_cluster_groups(G, resolution=config["LOUVAIN_RESOLUTION"], verbose=True)
        cluster_info = check_path_between_clusters(G, clusters)
        
        clusters_main_information = {ci: {"num_paths": 0, 
                                          "num_nodes": len(clusters[ci]), 
                                          "cluster_edges": [], 
                                          "l2_edge_paths": {}} for ci in clusters.keys()}
        
        # gather connectivity information between clusters
        for (cluster_a, cluster_b), (found_path, paths) in cluster_info.items():
            if found_path:        
                clusters_main_information[cluster_a]["num_paths"] += len(paths)
                clusters_main_information[cluster_a]["cluster_edges"].append(cluster_b)
                clusters_main_information[cluster_b]["num_paths"] += len(paths)              
                clusters_main_information[cluster_b]["cluster_edges"].append(cluster_a)
        
                for pth in paths:
                    if cluster_b not in clusters_main_information[cluster_a]["l2_edge_paths"]:
                        clusters_main_information[cluster_a]["l2_edge_paths"][cluster_b] = []
                    if cluster_a not in clusters_main_information[cluster_b]["l2_edge_paths"]:
                        clusters_main_information[cluster_b]["l2_edge_paths"][cluster_a] = []
        
                    clusters_main_information[cluster_b]["l2_edge_paths"][cluster_a].append([pth[1], pth[0]])
                    clusters_main_information[cluster_a]["l2_edge_paths"][cluster_b].append([pth[0], pth[1]])
        
        clusters_main_information, modified_clusters = merge_clusters(clusters_main_information, clusters, config["MAX_PATHS"], 
                                                                      config["NUM_EDGES"], config["MAX_PATH_PARTNER"], config["NUM_EDGES_PARTNER"])
        next_pairs = []
        print(f"Searching for paths that have {config['NUM_EDGES_PATHCUT']} edge cluster pair(s) and max {config['MAX_PATHS_PATHCUT']} paths to BREAK ...")
        for ci in clusters_main_information.keys():
            if clusters_main_information[ci]["num_paths"] <= config["MAX_PATHS_PATHCUT"] and len(clusters_main_information[ci]["cluster_edges"]) == config["NUM_EDGES_PATHCUT"]:
                next_pairs.append([ci, clusters_main_information[ci]])

        next_pairs = sorted(next_pairs, key=lambda d: d[1]['num_nodes'], reverse=True)    
        print(f"- number of potential pairs to break: {len(next_pairs)}\n")


        for p_index, pair in enumerate(next_pairs):
            cluster_a, cluster_a_information = pair[0], pair[1]
            cluster_a_num_nodes = cluster_a_information["num_nodes"]

            cluster_b = cluster_a_information['cluster_edges'][0]
            paths = cluster_a_information['l2_edge_paths'][cluster_b]
            
            # assert the l2 node ids are coming from the same parent root id
            all_l2_nodes = [node for pth in cluster_a_information["l2_edge_paths"][cluster_b] for node in pth]
            assert root_id == np.unique(client.chunkedgraph.get_roots(all_l2_nodes))[0]
            monster_root_id = root_id
            
            source_l2_nodes, sink_l2_nodes = [pth[0] for pth in paths], [pth[1] for pth in paths] # combine paths to source_l2_node and sink_l2_node
        
            print(f"{p_index+1}. Working with root id: {root_id} and {cluster_a}-{cluster_b} pair")
            print(f"- # of paths between {cluster_a}-{cluster_b}: {len(paths)}")
            print(f"- # of nodes in cluster: {cluster_a_num_nodes}")
            
            source_l2_node, sink_l2_node = source_l2_nodes[0], sink_l2_nodes[0]
            source_l2_nm, sink_l2_nm = l2stats[str(source_l2_node)]["rep_coord_nm"], l2stats[str(sink_l2_node)]["rep_coord_nm"]
            
            source_nm_pts, sink_nm_pts, source_l2_ids_in_use, sink_l2_ids_in_use = get_nm_pts(source_l2_nodes, sink_l2_nodes, modified_clusters, 
                                                                                              l2stats, G, use_neighbors=True, max_nodes=8)
            print(f"- # of source and sink pts: {len(source_nm_pts)}-{len(sink_nm_pts)}")
            # assert all l2 ids in use are from the same root id
            all_l2_nodes = [l2id for l2id in source_l2_ids_in_use+sink_l2_ids_in_use]
            assert root_id == np.unique(client.chunkedgraph.get_roots(all_l2_nodes))[0]

            source_pts_arr, sink_pts_arr = np.array(source_nm_pts), np.array(sink_nm_pts)
            source_max_dist, sink_max_dist = max_distance(source_pts_arr), max_distance(sink_pts_arr)
        
            if source_max_dist > config["L2_NODE_MAX_DIST"] or sink_max_dist > config["L2_NODE_MAX_DIST"]:
                print(f"- (WARNING) l2 nodes TOO FAR APART {source_max_dist},{sink_max_dist}. Skipping to next pair ...\n")
                continue
            else:
                print(f"- l2 nodes are in close proxmity! {source_max_dist} {sink_max_dist}.")

            SPLIT_MADE = False
            if config["PREVIEW"]:
                try: #For debugging purposes (preview mode)
                    results = do_preview_split(client, root_id, source_l2_node, sink_l2_node, source_nm_pts, sink_nm_pts)
                    operation_id, new_root_ids = -1, []
                    print(f"Split success?: {results[2]}")
                    source_super_voxel_list, sink_super_voxel_list = results[0], results[1]
                    SPLIT_MADE = True
                    print("\n")
                except Exception as e:
                    print(f"(ERROR) Preview ERROR occurred: {str(e)}")
            else:
                try: #split
                    operation_id, new_root_ids = do_split(client, root_id, source_nm_pts, sink_nm_pts)
                
                    # save results to json file
                    output_file = save_to_json(root_id, source_l2_node, sink_l2_node, 
                                source_l2_nm, sink_l2_nm, operation_id, new_root_ids, save_dir=splits_dir, verbose=True)
            
                    if len(new_root_ids) == 1: #special case
                        # not common for new roots created to only have one new id
                        print(f"- root ID split but no new root ids created!")
                        monster_root_id = new_root_ids[0]
                        new_smaller_root_ids = []
                        print(f"- new monster root id: {monster_root_id}")
                        print(f"- new smaller root ids created: {new_smaller_root_ids}")
                    else:
                        # switch to new monster segment id
                        monster_root_id, new_smaller_root_ids = find_monster_segment_id(client, new_root_ids, verbose=False)
                        print(f"- new monster root id: {monster_root_id}")
                        print(f"- new smaller root ids created: {new_smaller_root_ids}")
                    
                    SPLIT_MADE = True
                except Exception as e:
                    print(f"(ERROR) Split ERROR occurred: {str(e)}\n")
                    
            if SPLIT_MADE: # stop here IF split made
                break
            
            print("Trying the next pair ...")
            print("\n")
            
        
        print(f"SWITCHING root id from {root_id} -> {monster_root_id}\n")
        root_id = monster_root_id

        if ri == config["ROUNDS"]-1:
            print('done.')
            print(f'- left off with ~< {len(l2stats)} l2 nodes')
        else:
            print('waiting ...')
            time.sleep(config["WAIT_SEC"])
        

    # log_file.close()