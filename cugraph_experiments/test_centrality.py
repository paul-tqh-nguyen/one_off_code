
import cugraph
import cudf

# Helpers

# Katz Centrality

def test_katz_centrality_undirected():
    '''
 0 - 1    5 - 6
 | / |    | /
 3 - 4 -- 2 - 7
    '''
    sources =      [0, 0, 0, 1, 2, 2, 2, 2, 3, 5]
    destinations = [1, 3, 4, 4, 4, 5, 6, 7, 4, 6]
    assert len(sources) == 10
    assert len(destinations) == 10
    nodes = set(sources+destinations)
    assert len(nodes) == 8
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')
    
    assert g.number_of_edges() == 20 # one for both directions
    assert g.number_of_vertices() == 8
    assert g.number_of_nodes() == 8
    
    centrality_df = cugraph.katz_centrality(g)
    assert 'vertex' in centrality_df.columns
    assert 'katz_centrality' in centrality_df.columns
    node_keys = centrality_df['vertex']
    centralities = centrality_df['katz_centrality']
    sorted_centralities = sorted(list(zip(node_keys, centralities)), reverse=True, key=lambda x:x[1])
    assert sorted_centralities[0][0] == 4
    assert sorted_centralities[1][0] == 2
    assert sorted_centralities[2][0] == 0
    assert sorted_centralities[-1][0] == 7
