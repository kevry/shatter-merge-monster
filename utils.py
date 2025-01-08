import community as louvain
from copy import deepcopy
import json
import networkx as nx
import numpy as np
import os
from tqdm import tqdm


""" Helper functions to run main script """


def check_latest_root_id(client, latest_root_id, verbose=False, replace_root_id=False):
    """ Check if the root id being used is the latest root id in the system """
    is_latest = True
    if verbose:
        print(f"Confirming root id {latest_root_id} is the latest.")
    if client.chunkedgraph.is_latest_roots([latest_root_id])[0] == False:
        latest_root_ids = client.chunkedgraph.get_latest_roots(latest_root_id)
        is_latest = False

        if verbose:
            print("- root id is not the latest.")

        if replace_root_id:
            print("- finding/replacing with latest root id")
            latest_root_id = latest_root_ids[0]
            if verbose:
                print("- list of successor root ids:", latest_root_ids)
                print("- root id changed to ->", latest_root_id)
    else:
        if verbose:
            print("- Latest root id\n")
    return latest_root_id, is_latest


def is_monster_segment(client, root_id, verbose=False, thresh_n_nodes = 20000):
    """ test if the root id provided is a "monster root id" - very large object in segmentation """
    if verbose:
        print(f"Confirming root id {root_id} is a merge monster given thresh={thresh_n_nodes}.")
    l2nodes = client.chunkedgraph.get_leaves(root_id, stop_layer=2)
    if len(l2nodes) > thresh_n_nodes:
        if verbose:
            print("- root id is a monster segment")
            print(f"- with {len(l2nodes)} l2 nodes.\n")
        return True
    else:
        if verbose:
            print("- root id is NOT a monster segment")
            print(f"- with {len(l2nodes)} l2 nodes.\n")
        return False


def create_nx_graph(l2stats, lvl2_eg, verbose=False):
    """Create a NX graph using the L2 nodes in chunkedgraph
    Assumes l2stats has "rep_coord_nm" key available """
    G = nx.Graph() # create initial graph
    for i in range(len(lvl2_eg)):
        l2_idx1, l2_idx2 = lvl2_eg[i]
        euc_dist = np.linalg.norm(np.array(l2stats[str(l2_idx1)]['rep_coord_nm']) - np.array(l2stats[str(l2_idx2)]['rep_coord_nm'])) #euclidian distance between skeleton nodes ... in nm
        G.add_edge(l2_idx1, l2_idx2, weight=euc_dist)
    if verbose:
        print("Created graph!")
        print("Number of nodes in graph G:", G.number_of_nodes())
    return G


def get_root_id_l2_data(client, root_id, return_graph=True):
    """Get all information about l2cache for specific root id"""
    print("Gathering data from l2 cache:")

    lvl2nodes = client.chunkedgraph.get_leaves(root_id, stop_layer=2)
    print("- # of l2 nodes:", len(lvl2nodes))

    # get coordinates of these L2 nodes
    l2stats=client.l2cache.get_l2data(lvl2nodes, attributes=['rep_coord_nm', 'size_nm3'])
        
    for key, val in l2stats.items(): # check no data is missing
        if val == {}:
            print(f"(Error) missing data for l2 node {key}. Wait while l2 cache is updating ...")
            return None, None, None, False
        
    print("- done. All l2 cache data collected.\n")
    lvl2_eg = client.chunkedgraph.level2_chunk_graph(root_id) # get edges
    if return_graph:
        G = create_nx_graph(l2stats, lvl2_eg, verbose=False)
        return l2stats, lvl2_eg, G, True
    else:
        return l2stats, lvl2_eg, _, True
    

def create_cluster_groups(G, resolution = 20, verbose=False):
    """ Create cluster groups using louvain method """

    if verbose:
        print(f"Creating community clusters w/ louvain and w/ resolution={resolution}")
    partition = louvain.best_partition(G, resolution=resolution)

    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)

    if verbose:
        print(f"- total number of clusters created: {len(clusters.keys())}\n")

    return clusters


def check_path_between_clusters(G, clusters):
    """ Check if there is a path between single groups"""

    print("Check connectivity between clusters.")
    connectivity = {}
    cluster_ids = list(clusters.keys())

    for i in range(len(cluster_ids)): #tqdm(range(len(cluster_ids)), desc="Collecting connectivity info"):
        for j in range(i + 1, len(cluster_ids)):
            cluster_i = clusters[cluster_ids[i]].copy()
            cluster_j = clusters[cluster_ids[j]].copy()

            nodes_combined = cluster_i + cluster_j

            found_path = False
            # Check for any path between nodes in different clusters
            paths_from_i_to_j = []
            for node_i in cluster_i:
                n_neighbors = G.neighbors(node_i) # check each nodes neighbors
                for nn in n_neighbors:
                    if nn in cluster_j: # if neighbor outside cluster
                        paths_from_i_to_j.append([node_i, nn])
                        found_path = True

            connectivity[(cluster_ids[i], cluster_ids[j])] = [found_path, paths_from_i_to_j]

    print("- done.\n")
    return connectivity


def merge_clusters(clusters_main_information, clusters, MAX_PATHS, NUM_EDGES, MAX_PATH_PARTNER, NUM_EDGES_PARTNER):
    """ Merge small clusters together """

    print("Merging small clusters together:")

    # find clusters that have a max of {MAX_PATHS} paths to other clusters and connected to ONLY {NUM_EDGES} clusters ??
    single_path_cluster_ids = [ci for ci in clusters_main_information.keys() if clusters_main_information[ci]["num_paths"] <= MAX_PATHS and len(clusters_main_information[ci]["cluster_edges"]) == NUM_EDGES]
    
    merged_clusters = []
    
    # create deepcopy of cluster dict to modify along the way
    modified_clusters = deepcopy(clusters)
    
    # find clusters with {NUM_EDGES_PARTNER} edges to other clusters and less than {MAX_PATH_PARTNER} paths
    for ci in clusters_main_information.keys():
        if clusters_main_information[ci]["num_paths"] <= MAX_PATH_PARTNER and len(clusters_main_information[ci]["cluster_edges"]) == NUM_EDGES_PARTNER:
            for cluster_edge_id in clusters_main_information[ci]["cluster_edges"]:
                if cluster_edge_id in single_path_cluster_ids:
                    print(f"- merging {cluster_edge_id} with {ci}")
                    modified_clusters[ci].extend(clusters[cluster_edge_id].copy())
                    del modified_clusters[cluster_edge_id]
                    merged_clusters.append({"root_cluster": ci, "merged_cluster": cluster_edge_id})
    
    # remove from main cluster information
    for mc in merged_clusters:
        root_cluster_id, merged_cluster_id = mc["root_cluster"], mc["merged_cluster"]
        clusters_main_information[root_cluster_id]["cluster_edges"].remove(merged_cluster_id)
        clusters_main_information[root_cluster_id]["num_nodes"] += clusters_main_information[merged_cluster_id]["num_nodes"]
        clusters_main_information[root_cluster_id]["num_paths"] -= clusters_main_information[merged_cluster_id]["num_paths"]
        clusters_main_information[root_cluster_id]["l2_edge_paths"].pop(merged_cluster_id, None)
        clusters_main_information.pop(merged_cluster_id, None)

    print("- done.\n")
    return clusters_main_information, modified_clusters

def get_nm_pts(source_l2_nodes, sink_l2_nodes, modified_clusters, l2stats, G, use_neighbors = False, max_nodes=100000):
    print("- getting nm coordinates for source and sink nodes ...")
    source_nm_pts = []
    sink_nm_pts = []

    cluster_2_map = create_l2_cluster_mappings(modified_clusters)

    # assert l2 source and sink nodes are from different clusters
    assert len(np.unique([cluster_2_map[l2id] for l2id in source_l2_nodes])) == 1 
    assert len(np.unique([cluster_2_map[l2id] for l2id in sink_l2_nodes])) == 1
    assert cluster_2_map[source_l2_nodes[0]] != cluster_2_map[sink_l2_nodes[0]]

    # get l2 node source and sink nm coordinates
    source_nodes_in_use, sink_nodes_in_use = [],[]
    for l2id in source_l2_nodes:
        source_nm_pts.append(l2stats[str(l2id)]["rep_coord_nm"])
        source_nodes_in_use.append(l2id)
    for l2id in sink_l2_nodes:
        sink_nm_pts.append(l2stats[str(l2id)]["rep_coord_nm"])
        sink_nodes_in_use.append(l2id)
    
    
    if use_neighbors: #get neighbors of source and sink
        source_neighbors = {}
        for l2id in source_l2_nodes:
            source_neighbors[l2id] = []
            for n in G.neighbors(l2id):
                source_neighbors[l2id].append(n)

        sink_neighbors = {}
        for l2id in sink_l2_nodes:
            sink_neighbors[l2id] = []
            for n in G.neighbors(l2id):
                sink_neighbors[l2id].append(n)

        
        source_done = False
        sink_done = False
        while source_done == False and sink_done == False:

            # source search
            source_l2_node_complete_count = 0
            for l2id in source_l2_nodes:
                potential_ns = source_neighbors[l2id]
                if potential_ns == []:
                    source_l2_node_complete_count += 1
                    continue
                else:
                    pn = potential_ns[0]
                    source_neighbors[l2id].pop(0) # remove after used
                    if pn not in source_nodes_in_use:
                        if cluster_2_map[l2id] == cluster_2_map[pn]: # assert neighbor node is in same cluster group
                            source_nm_pts.append(l2stats[str(pn)]["rep_coord_nm"])
                            source_nodes_in_use.append(pn)

                            if len(source_nm_pts) >= max_nodes:
                                source_done = True
                                break
                            
            if len(source_l2_nodes) == source_l2_node_complete_count:
                source_done = True
                break # exit so both source and sink pt count are equal

            # sink search
            sink_l2_node_complete_count = 0
            for l2id in sink_l2_nodes:
                potential_ns = sink_neighbors[l2id]
                if potential_ns == []:
                    sink_l2_node_complete_count += 1
                    continue
                else:
                    pn = potential_ns[0]
                    sink_neighbors[l2id].pop(0) # remove after used
                    if pn not in sink_nodes_in_use:
                        if cluster_2_map[l2id] == cluster_2_map[pn]: # asset neighbor node is in same cluster group
                            sink_nm_pts.append(l2stats[str(pn)]["rep_coord_nm"])
                            sink_nodes_in_use.append(pn)

                            if len(sink_nm_pts) >= max_nodes:
                                sink_done = True
                                break
                            
            if len(sink_l2_nodes) == sink_l2_node_complete_count: # if no more neighbors to use ... exit
                sink_done = True
                break

    return source_nm_pts, sink_nm_pts, source_nodes_in_use, sink_nodes_in_use


def find_monster_segment_id(client, new_root_ids, verbose=False):
    """ using new root ids created after splitting, find which is still the monster merge error """
    monster_root_id, max_l2_nodes = 0, 0
    for rid in new_root_ids:
        l2nodes = client.chunkedgraph.get_leaves(rid, stop_layer=2) # get number of l2 nodes for root id
        if len(l2nodes) > max_l2_nodes:
            monster_root_id = rid
            max_l2_nodes = len(l2nodes)

    # make these the new smaller root ids
    new_smaller_root_ids = new_root_ids.copy()
    new_smaller_root_ids.remove(monster_root_id)

    if verbose:
        print(f"Monster root ID is {monster_root_id} with {max_l2_nodes} l2 nodes.")
        print("New smaller root ids:", new_smaller_root_ids)
        
    return monster_root_id, new_smaller_root_ids

    
def do_preview_split(client, root_id, source_l2_node, sink_l2_node, source_nm_pts, sink_nm_pts, super_voxels_source = [], super_voxels_sink = []):
    """ split preview for testing """
    print("RUNNING PREVIEW.")
    #print("Number of source-sink pts:", len(source_nm_pts), len(sink_nm_pts))

    results = client.chunkedgraph.preview_split(source_nm_pts, sink_nm_pts, 
                                                root_id=root_id)
    return results


def do_split(client, root_id, source_nm_pts, sink_nm_pts, source_supervoxels = [], sink_supervoxels = []):
    """ execute splitting """
    print("EXECUTING SPLIT.")
    #print("Number of source-sink pts:", len(source_nm_pts), len(sink_nm_pts))

    results = client.chunkedgraph.execute_split(source_nm_pts, sink_nm_pts, 
                                                root_id=root_id,)
    operation_id, new_root_ids = results
    return operation_id, new_root_ids


def create_l2_cluster_mappings(modified_clusters):
    """ Use the clusters dict to create a l2id->clusterid mapping """
    cluster_l2_id_map = {} # create l2nodeid -> cluster id mapping
    for ci in modified_clusters.keys():
        for n in modified_clusters[ci]:
            cluster_l2_id_map[n] = ci
    return cluster_l2_id_map


def save_to_json(root_id, source_l2_node, sink_l2_node, source_l2_nm, sink_l2_nm, operation_id, new_root_ids, save_dir, verbose=False):
    """ save split results to json for book-keeping ... """
    split_information = {
        "operation_id": operation_id, 
        "new_root_ids": new_root_ids, 
        "root_id": root_id, 
        "l2_source_node_id": source_l2_node, 
        "l2_sink_node_id": sink_l2_node,
        "l2_source_node_nm_coord": source_l2_nm, 
        "l2_sink_node_nm_coord": sink_l2_nm
    }
    output_file = os.path.join(save_dir, f"split_{root_id}_{operation_id}.json")
    with open(output_file, "w") as f:
        json.dump(split_information, f)
    if verbose:
        print(f"Saved {output_file}")
    return output_file
        

def max_distance(points):
    max_dist = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


class OutBoth:
    def __init__(self, *files):
        self.files = files
   
    def write(self, message):
        for file in self.files:
            file.write(message)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()



def find_monster(client, root_id):
    """ find merge monster using an initial root id """
    found_monster = False
    original_root_id = root_id
    
    while not found_monster:
        potential_latest_root_id = client.chunkedgraph.get_latest_roots(root_id)
        if len(potential_latest_root_id) == 1: 
            if potential_latest_root_id[0] == root_id:
                n_l2_nodes = len(client.chunkedgraph.get_leaves(root_id, stop_layer=2))
                print(f"{root_id} is the latest merge monster with {n_l2_nodes} nodes.")
            else:
                n_l2_nodes = len(client.chunkedgraph.get_leaves(root_id, stop_layer=2))
                print(f"{root_id} changed to {potential_latest_root_id[0]} with {n_l2_nodes} nodes.")
                root_id = potential_latest_root_id[0]
            found_monster = True
        else:
            for pid in potential_latest_root_id:
                n_l2_nodes = len(client.chunkedgraph.get_leaves(pid, stop_layer=2))
                if n_l2_nodes > 20000:
                    print(f"{root_id} -> {pid}. New potential monster with {n_l2_nodes} nodes.")
                    root_id = pid
                    break
    return root_id