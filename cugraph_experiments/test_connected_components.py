
import cugraph
import cudf

# Connected Components

# Strongly Connected Components

def test_strongly_connected_components_unweighted_undirected():
    '''
 0 - 1    5 - 6
 | X |    | /
 3 - 4 -- 2 - 7
    '''
    sources =      [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 5]
    destinations = [1, 3, 4, 3, 4, 4, 5, 6, 7, 4, 6]
    assert len(sources) == 11
    assert len(destinations) == 11
    nodes = set(sources+destinations)
    assert len(nodes) == 8
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')
    
    assert g.number_of_edges() == 22 # one for both directions
    assert g.number_of_vertices() == 8
    assert g.number_of_nodes() == 8
    
    strongly_connected_components_df = cugraph.strongly_connected_components(g)
    # All connected undirected graphs are strongly connected since an undirected edge goes both directions
    assert strongly_connected_components_df['labels'].nunique() == 1

def test_strongly_connected_components_disconnected_unweighted_undirected():
    '''
 0 - 1    5 - 6
 | /      | /
 3   4 -- 2 - 7
    '''
    sources =      [0, 0, 1, 2, 2, 2, 2, 6]
    destinations = [1, 3, 3, 4, 5, 6, 7, 7]
    assert len(sources) == 8
    assert len(destinations) == 8
    nodes = set(sources+destinations)
    assert len(nodes) == 8
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')
    
    assert g.number_of_edges() == 16 # one for both directions
    assert g.number_of_vertices() == 8
    assert g.number_of_nodes() == 8
    
    strongly_connected_components_df = cugraph.strongly_connected_components(g)
    assert strongly_connected_components_df['labels'].nunique() == 2 # @todo the labels are not in a contiguous sequence.
    strongly_connected_components_df_groupby = strongly_connected_components_df.groupby(['labels'])
    assert strongly_connected_components_df_groupby.min()['vertices'].sum() == 0 + 2
    assert strongly_connected_components_df_groupby.min()['vertices'].product() == 0 * 2
    assert strongly_connected_components_df_groupby.min()['vertices'].sum_of_squares() == 0**2 + 2**2
    assert strongly_connected_components_df_groupby.max()['vertices'].sum() == 3 + 7
    assert strongly_connected_components_df_groupby.max()['vertices'].product() == 3 * 7
    assert strongly_connected_components_df_groupby.max()['vertices'].sum_of_squares() == 3**2 + 7**2
    assert strongly_connected_components_df_groupby.count()['vertices'].sum() == 3 + 5
    assert strongly_connected_components_df_groupby.count()['vertices'].product() == 3 * 5
    assert strongly_connected_components_df_groupby.count()['vertices'].sum_of_squares() == 3**2 + 5**2

def test_strongly_connected_components_unweighted_directed():
    '''
0 < -   1       5   - > 6
      ^       ^ ^       
|   /   |   /   |   /    
v       v /       v      
3   - > 4 < -   2   - > 7
    '''
    sources =      [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    assert len(sources) == 11
    assert len(destinations) == 11
    nodes = set(sources+destinations)
    assert len(nodes) == 8
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')
    
    assert g.number_of_edges() == 11
    assert g.number_of_vertices() == 8
    assert g.number_of_nodes() == 8
    
    strongly_connected_components_df = cugraph.strongly_connected_components(g)
    # All connected undirected graphs are strongly connected since an undirected edge goes both directions
    assert strongly_connected_components_df['labels'].nunique() == 3 # (0,1,3) (2,4,5,6) (7)
    strongly_connected_components_df_groupby = strongly_connected_components_df.groupby(['labels'])
    assert strongly_connected_components_df_groupby.min()['vertices'].sum() == 0 + 2 + 7
    assert strongly_connected_components_df_groupby.min()['vertices'].product() == 0 * 2 * 7
    assert strongly_connected_components_df_groupby.min()['vertices'].sum_of_squares() == 0**2 + 2**2 + 7**2
    assert strongly_connected_components_df_groupby.max()['vertices'].sum() == 3 + 6 + 7
    assert strongly_connected_components_df_groupby.max()['vertices'].product() == 3 * 6 * 7
    assert strongly_connected_components_df_groupby.max()['vertices'].sum_of_squares() == 3**2 + 6**2 + 7**2
    assert strongly_connected_components_df_groupby.count()['vertices'].sum() == 3 + 4 +1
    assert strongly_connected_components_df_groupby.count()['vertices'].product() == 3 * 4 * 1
    assert strongly_connected_components_df_groupby.count()['vertices'].sum_of_squares() == 3**2 + 4**2 + 1**2

# Weakly Connected Components

def test_weakly_connected_components_unweighted_undirected():
    '''
 0 - 1    5 - 6
 | X |    | /
 3 - 4 -- 2 - 7
    '''
    sources =      [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 5]
    destinations = [1, 3, 4, 3, 4, 4, 5, 6, 7, 4, 6]
    assert len(sources) == 11
    assert len(destinations) == 11
    nodes = set(sources+destinations)
    assert len(nodes) == 8
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')
    
    assert g.number_of_edges() == 22 # one for both directions
    assert g.number_of_vertices() == 8
    assert g.number_of_nodes() == 8
    
    weakly_connected_components_df = cugraph.weakly_connected_components(g)
    # All connected undirected graphs are weakly connected since an undirected edge goes both directions
    assert weakly_connected_components_df['labels'].nunique() == 1

def test_weakly_connected_components_disconnected_unweighted_undirected():
    '''
 0 - 1    5 - 6
 | /      | /
 3   4 -- 2 - 7
    '''
    sources =      [0, 0, 1, 2, 2, 2, 2, 6]
    destinations = [1, 3, 3, 4, 5, 6, 7, 7]
    assert len(sources) == 8
    assert len(destinations) == 8
    nodes = set(sources+destinations)
    assert len(nodes) == 8
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')
    
    assert g.number_of_edges() == 16 # one for both directions
    assert g.number_of_vertices() == 8
    assert g.number_of_nodes() == 8
    
    weakly_connected_components_df = cugraph.weakly_connected_components(g)
    assert weakly_connected_components_df['labels'].nunique() == 2 # @todo the labels are not in a contiguous sequence.
    weakly_connected_components_df_groupby = weakly_connected_components_df.groupby(['labels'])
    assert weakly_connected_components_df_groupby.min()['vertices'].sum() == 0 + 2
    assert weakly_connected_components_df_groupby.min()['vertices'].product() == 0 * 2
    assert weakly_connected_components_df_groupby.min()['vertices'].sum_of_squares() == 0**2 + 2**2
    assert weakly_connected_components_df_groupby.max()['vertices'].sum() == 3 + 7
    assert weakly_connected_components_df_groupby.max()['vertices'].product() == 3 * 7
    assert weakly_connected_components_df_groupby.max()['vertices'].sum_of_squares() == 3**2 + 7**2
    assert weakly_connected_components_df_groupby.count()['vertices'].sum() == 3 + 5
    assert weakly_connected_components_df_groupby.count()['vertices'].product() == 3 * 5
    assert weakly_connected_components_df_groupby.count()['vertices'].sum_of_squares() == 3**2 + 5**2

def test_weakly_connected_components_unweighted_directed():
    '''
0 < -   1       5   - > 6
      ^       ^ ^       
|   /   |   /   |   /    
v       v /       v      
3   - > 4 < -   2   - > 7
    '''
    sources =      [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    assert len(sources) == 11
    assert len(destinations) == 11
    nodes = set(sources+destinations)
    assert len(nodes) == 8
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')
    
    assert g.number_of_edges() == 11
    assert g.number_of_vertices() == 8
    assert g.number_of_nodes() == 8
    
    weakly_connected_components_df = cugraph.weakly_connected_components(g)
    assert weakly_connected_components_df['labels'].nunique() == 3 # (0,1,3) (2,4,5,6) (7)
    weakly_connected_components_df_groupby = weakly_connected_components_df.groupby(['labels'])
    assert weakly_connected_components_df_groupby.min()['vertices'].sum() == 0 + 2 + 7
    assert weakly_connected_components_df_groupby.min()['vertices'].product() == 0 * 2 * 7
    assert weakly_connected_components_df_groupby.min()['vertices'].sum_of_squares() == 0**2 + 2**2 + 7**2
    assert weakly_connected_components_df_groupby.max()['vertices'].sum() == 3 + 6 + 7
    assert weakly_connected_components_df_groupby.max()['vertices'].product() == 3 * 6 * 7
    assert weakly_connected_components_df_groupby.max()['vertices'].sum_of_squares() == 3**2 + 6**2 + 7**2
    assert weakly_connected_components_df_groupby.count()['vertices'].sum() == 3 + 4 +1
    assert weakly_connected_components_df_groupby.count()['vertices'].product() == 3 * 4 * 1
    assert weakly_connected_components_df_groupby.count()['vertices'].sum_of_squares() == 3**2 + 4**2 + 1**2

def test_asdf():
    '''
0 < -   2
      ^  
|   /
v
1
    '''
    # sources =      [0, 1, 2, 1, 2, 0]
    # destinations = [1, 2, 0, 0, 1, 2]
    sources =      [0, 1, 2, 1]
    destinations = [1, 2, 0, 0]
    
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(gdf, source='source', destination='destination')

    print(f'gdf {gdf}')
    
    weakly_connected_components_df = cugraph.weakly_connected_components(g)
