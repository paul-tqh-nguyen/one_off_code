
const hierarchyMain = () => {
    
    const dataLocation = './hierarchy_data.json';

    const mean = inputArray => inputArray.reduce((a, b) => a + b, 0) / inputArray.length;
    const shuffle = (inputArray) => inputArray.sort(() => Math.random() - 0.5);

    const plotContainer = document.getElementById('hierarchy');
    const svg = d3.select('#hierarchy-svg');
    
    const alphaDecay = 0.01;
    const velocityDecay = 0.00001;
    const distanceToCenterAlpha = 1.0;
    const linkAlpha = 0.5;
    const siblingAlpha = 0.5;

    const paddingBetweenNodes = 15;
    const approximateCircumferenceDistancePerNode = 15;
    const minDistanceBetweenDepths = 200;
    const margin = {
        top: 100,
        bottom: 100,
        left: 100,
        right: 100,
    };
    const textDisplayX = 30;
    const textDisplayY = 20;
    const textDisplayTSPANHeight = '1.3em';
    const textDisplayMargin = {
        top: 10,
        bottom: 10,
        left: 10,
        right: 10,
    };
    const textDisplayBackgroundColor = '#fff';
    const textDisplayBorderColor = 'purple';
    const textDisplayBorderWidth = 1;

    const simulation = d3.forceSimulation()
	  .alphaDecay(alphaDecay)
	  .velocityDecay(velocityDecay);
    
    const render = ({nodeData, linkData, rootNode, nodeById, parentIdToChildIds, childIdToParentids, distanceToCenterFactorByDepth}) => {
	
	svg
	    .attr('width', `${plotContainer.clientWidth}px`)
	    .attr('height', `${plotContainer.clientHeight}px`)
	    .selectAll('*')
	    .remove();
	const svgWidth = parseFloat(svg.attr('width'));
	const svgHeight = parseFloat(svg.attr('height'));
                
        const svgZoomableContent = svg.append('g');
        svg.call(d3.zoom().on('zoom', () => {
            svgZoomableContent.attr('transform', d3.event.transform);
        }));
	const edgeGroup = svgZoomableContent.append('g')
	      .selectAll('line')
	      .data(linkData)
	      .enter()
              .append('line')
	      .attr('class', datum => nodeById[datum.child].distance_to_root - nodeById[datum.parent].distance_to_root > 1 ? 'indirect-edge' : 'direct-edge')
              .on('mouseover', function(datum) {
                  if (d3.select(this).classed('direct-edge')) {
                      d3.select(this).attr('class', 'direct-edge direct-edge-highlighted');
                  }
                  if (d3.select(this).classed('indirect-edge')) {
                      d3.select(this).attr('class', 'indirect-edge indirect-edge-highlighted');
                  }
                  const parent = nodeById[datum.parent];
                  const child = nodeById[datum.child];
                  d3.select('#text-display')
                      .html(`
<p>Parent:</p>
<p>Label: ${parent.label} </p>
<p>Description: ${parent.description} </p>
<p>Number of Instances: ${parent.number_of_instances} </p>
<p>Wikidata ID: <a target="_blank" title="${parent.label}"href="https://www.wikidata.org/wiki/${parent.id.replace('wd:','')}">${parent.id}</a></p>
</br></br></br></br></br>
<p>Child:</p>
<p>Label: ${child.label} </p>
<p>Description: ${child.description} </p>
<p>Number of Instances: ${child.number_of_instances} </p>
<p>Wikidata ID: <a target="_blank" title="${child.label}"href="https://www.wikidata.org/wiki/${child.id.replace('wd:','')}">${child.id}</a></p>
`,);
              })
              .on('mouseout', function(d) {
                  if (d3.select(this).classed('direct-edge direct-edge-highlighted')) {
                      d3.select(this).attr('class', 'direct-edge');
                  }
                  if (d3.select(this).classed('indirect-edge indirect-edge-highlighted')) {
                      d3.select(this).attr('class', 'indirect-edge');
                  }
              });

	const nodeGroup = svgZoomableContent.append('g')
	      .selectAll('circle')
	      .data(nodeData)
	      .enter().append('circle')
              .attr('class', 'node')
              .on('mouseover', datum => {
                  d3.select('#text-display')
                      .html(`
<p>Label: ${datum.label} </p>
<p>Description: ${datum.description} </p>
<p>Number of Instances: ${datum.number_of_instances} </p>
<p>Wikidata ID: <a target="_blank" title="${datum.label}"href="https://www.wikidata.org/wiki/${datum.id.replace('wd:','')}">${datum.id}</a></p>
`,);
              })
              .on('click', datum => {
                  console.log(`datum.label ${JSON.stringify(datum.label)}`); // @todo do something here
              });

        const distanceToCenter = alpha => {
            return () => {
	        nodeData.forEach(datum => {
                    if (datum !== rootNode) {
                        const goalDistance = distanceToCenterFactorByDepth[datum.distance_to_root];
                        const xDelta = rootNode.x - datum.x;
                        const yDelta = rootNode.y - datum.y;
                        const currentDistance = Math.sqrt(xDelta * xDelta + yDelta * yDelta);
                        const oldPortionToKeep = 1 - ((currentDistance - goalDistance) / currentDistance) * alpha;
                        datum.x = datum.x * oldPortionToKeep + (1-oldPortionToKeep) * rootNode.x;
		        datum.y = datum.y * oldPortionToKeep + (1-oldPortionToKeep) * rootNode.y;
	            }
                });
            };
        };
        
        const linkForce = alpha => {            
            return () => {
	        nodeData.forEach(child => {
                    if (child !== rootNode) {
		        const parentIds = childIdToParentids[child.id];
                        const parents = parentIds.map(parentId => nodeById[parentId]).filter(parent => (child.distance_to_root - parent.distance_to_root) == 1);
                        const parentMeanX = mean(parents.map(parent => parent.x));
                        const parentMeanY = mean(parents.map(parent => parent.y));
                        child.x = child.x * (1-alpha) + alpha * parentMeanX;
                        child.y = child.y * (1-alpha) + alpha * parentMeanY;
                    }
                });
            };
        };
        
        const siblingForce = alpha => {            
            return () => {
	        nodeData.forEach(parent => {
                    const siblings = parentIdToChildIds[parent.id]
                          .map(childId => childIdToParentids[childId])
                          .reduce((a,b) => a.concat(b), [])
                          .filter(siblingId => siblingId !== parent.id)
                          .map(siblingId => nodeById[siblingId])
                          .filter(sibling => sibling.distance_to_root == parent.distance_to_root);
                    if (siblings.length > 0) {
                        const siblingMeanX = mean(siblings.map(sibling => sibling.x)); 
                        const siblingMeanY = mean(siblings.map(sibling => sibling.y));
                        parent.x = parent.x * (1-alpha) + alpha * siblingMeanX;
                        parent.y = parent.y * (1-alpha) + alpha * siblingMeanY;
                    }
                });
            };
        };
        
	simulation
            .force('center', d3.forceCenter(svgWidth / 2, svgHeight / 2))
            .force('links', linkForce(linkAlpha))
            .force('sibling-force', siblingForce(siblingAlpha))
            .force('distance-to-center', distanceToCenter(distanceToCenterAlpha))
            .force('collide', d3.forceCollide(paddingBetweenNodes).strength(1))
	    .nodes(nodeData).on('tick', () => {
		nodeGroup
		    .attr('cx', datum => datum.x)
		    .attr('cy', datum => datum.y);
		edgeGroup
		    .attr('x1', datum => nodeById[datum.parent].x)
		    .attr('y1', datum => nodeById[datum.parent].y)
		    .attr('x2', datum => nodeById[datum.child].x)
		    .attr('y2', datum => nodeById[datum.child].y);
	    })
	    .restart();
    };

    const generateDistanceToCenterFactorByDepth = nodeData => {
        const nodesPerDepth = nodeData.reduce((accumulator, node) => {
            if (node.distance_to_root in accumulator) {
                accumulator[node.distance_to_root] += 1;
            } else {
                accumulator[node.distance_to_root] = 1;
            }
            return accumulator;
        }, {});
        const distanceToCenterFactorByDepth = Object.keys(nodesPerDepth).reduce((accumulator, depth) => {
            const nodeCount = nodesPerDepth[depth];
            const approximateCircumference = nodeCount * approximateCircumferenceDistancePerNode;
            const expectedRadius = approximateCircumference / (2 * Math.PI);
            accumulator[depth] = Math.max(minDistanceBetweenDepths, expectedRadius);
            return accumulator;
        }, {});
        Object.keys(distanceToCenterFactorByDepth).sort().reduce((currentDistanceFromRoot, depth) => {
            distanceToCenterFactorByDepth[depth] += currentDistanceFromRoot;
            currentDistanceFromRoot = distanceToCenterFactorByDepth[depth];
            return currentDistanceFromRoot;
        }, 0);
        return distanceToCenterFactorByDepth;
    };
    
    d3.json(dataLocation)
	.then(data => {
	    const nodeData = data.nodes;
	    const linkData = data.links;
            const rootNode = nodeData.filter(datum => datum.distance_to_root == 0)[0];
	    const nodeById = nodeData.reduce((accumulator, node) => {
		accumulator[node.id] = node;
                return accumulator;
            }, {});
	    const { parentIdToChildIds, childIdToParentids } = linkData.reduce((accumulator, datum) => {
                ['parent', 'child'].forEach(datumKey => {
                    ['parentIdToChildIds', 'childIdToParentids'].forEach(accumulatorKey => {
                        if (! (datum[datumKey] in accumulator[accumulatorKey]) ) {
		            accumulator[accumulatorKey][datum[datumKey]] = [];
                        }
                    });
                });
	    	accumulator.parentIdToChildIds[datum.parent].push(datum.child);
	    	accumulator.childIdToParentids[datum.child].push(datum.parent);
                return accumulator;
	    }, {parentIdToChildIds: {}, childIdToParentids: {}});
            const distanceToCenterFactorByDepth = generateDistanceToCenterFactorByDepth(nodeData);
            const redraw = () => render({
                nodeData,
                linkData,
                rootNode,
                nodeById,
                parentIdToChildIds,
                childIdToParentids,
                distanceToCenterFactorByDepth
            });
	    redraw();
	    window.addEventListener('resize', redraw);
	}).catch(err => {
	    console.error(err.message);
	    return;
	});
};

hierarchyMain();
