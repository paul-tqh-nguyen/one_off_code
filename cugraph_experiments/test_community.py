import cugraph
import cudf

def test_triangle_count_trivial():
    sources =      [0,1,2]
    destinations = [1,2,0]
    gdf = cudf.DataFrame({'source': sources, 'destination': destinations})
    g = cugraph.Graph()
    g.from_cudf_edgelist(gdf, source="source", destination="destination")
    assert cugraph.triangles(g) == 1


