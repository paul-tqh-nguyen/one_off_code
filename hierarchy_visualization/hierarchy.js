
let redraw = () => {};

const hierarchyMain = (dataLocationBaseName) => {

    const dataLocation = `./${dataLocationBaseName}_data.json`;
    
    // const shuffle = (inputArray) => inputArray.sort(() => Math.random() - 0.5);
    const sum = inputArray => inputArray.reduce((a, b) => a + b, 0);
    const mean = inputArray => sum(inputArray) / inputArray.length;
    // const softmax = (inputArray) => inputArray.map(value  => Math.exp(value) / sum(inputArray.map(y =>  Math.exp(y))));
    // const normalize = (inputArray) => inputArray.map(value  => value / sum(inputArray));
    // const simpleStringHash = inputString => inputString.split('').map(character => character.charCodeAt(0)).reduce((a,b) => parseInt(JSON.stringify(a)+JSON.stringify(b)));
    // const simpleStringToFloatHash = inputString => inputString.split('').map(character => character.charCodeAt(0)).reduce((a,b) => a/b);

    const plotContainer = document.getElementById('hierarchy');
    const svg = d3.select('#hierarchy-svg');
    d3.select('#text-display').html('');
    
    const alphaDecay = 0.001;
    const velocityDecay = 0.1;
    const distanceToCenterAlpha = 1.0;
    const linkAlpha = 0.1;
    const siblingAlpha = 0.25;

    const paddingBetweenNodes = 10;
    const approximateCircumferenceDistancePerNode = 10;
    const minDistanceBetweenDepths = 100;

    const simulation = d3.forceSimulation()
	  .alphaDecay(alphaDecay)
	  .velocityDecay(velocityDecay);
    const drag = d3.drag();
    
    const render = (inputArgs) => {

        const {nodeData, linkData, rootNode, nodeById, parentIdToChildIds, childIdToParentids, distanceToCenterFactorByDepth} = inputArgs;
        
        d3.select('#text-display').html(`
<p>Label: ${rootNode.label} </p>
<p>Description: ${rootNode.description} </p>
<p>Number of Subclasses: ${parentIdToChildIds[rootNode.id].length} </p>
<p>Number of Instances: ${rootNode.number_of_instances} </p>
<p>Wikidata ID: <a target="_blank" title="${rootNode.label}"href="https://www.wikidata.org/wiki/${rootNode.id.replace('wd:','')}">${rootNode.id}</a></p>
`,);
        
        svg
	    .attr('width', `${0}px`)
	    .attr('height', `${0}px`)
	    .attr('width', `${plotContainer.clientWidth}px`)
	    .attr('height', `${plotContainer.clientHeight}px`)
	    .selectAll('*')
	    .remove();
        
	const svgWidth = parseFloat(svg.attr('width'));
	const svgHeight = parseFloat(svg.attr('height'));

        nodeData.forEach((datum, index) => {
            if ( !('x' in datum && 'y' in datum)) {
                switch(index % 4) {
                case 0:
                    datum.x = 0;
                    datum.y = (index+1) / nodeData.length * svgHeight;
                    break;
                case 1:
                    datum.x = svgWidth;
                    datum.y = (index+1) / nodeData.length * svgHeight;
                    break;
                case 2:
                    datum.x = (index+1) / nodeData.length * svgWidth;
                    datum.y = 0;
                    break;
                case 3:
                    datum.x = (index+1) / nodeData.length * svgWidth;
                    datum.y = svgHeight;
                    break;
                }
            }
        });
                
        const svgZoomableContent = svg.append('g');
        svg.call(d3.zoom().on('zoom', () => {
            svgZoomableContent
                .attr('transform', d3.event.transform);
        }));
	const edgeGroup = svgZoomableContent.append('g')
	      .selectAll('line')
	      .data(linkData.filter(datum => nodeById[datum.parent].display_endabled && nodeById[datum.child].display_endabled))
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
<p>Number of Subclasses: ${parentIdToChildIds[parent.id].length} </p>
<p>Number of Instances: ${parent.number_of_instances} </p>
<p>Wikidata ID: <a target="_blank" title="${parent.label}"href="https://www.wikidata.org/wiki/${parent.id.replace('wd:','')}">${parent.id}</a></p>
</br>
<p>Child:</p>
<p>Label: ${child.label} </p>
<p>Description: ${child.description} </p>
<p>Number of Subclasses: ${parentIdToChildIds[child.id].length} </p>
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
	      .data(nodeData.filter(datum => datum.display_endabled))
	      .enter().append('circle')
              .attr('class', datum => parentIdToChildIds[datum.id].filter(childId => nodeById[childId].distance_to_root - datum.distance_to_root == 1).length > 0 ? 'node node-expandable' : 'node node-leaf')
              .on('mouseover', datum => {
                  d3.select('#text-display')
                      .html(`
<p>Label: ${datum.label} </p>
<p>Description: ${datum.description} </p>
<p>Number of Subclasses: ${parentIdToChildIds[datum.id].length} </p>
<p>Number of Instances: ${datum.number_of_instances} </p>
<p>Wikidata ID: <a target="_blank" title="${datum.label}"href="https://www.wikidata.org/wiki/${datum.id.replace('wd:','')}">${datum.id}</a></p>
`,);
              })
              .on('click', datum => {
                  const xDelta = datum.x - rootNode.x;
                  const yDelta = datum.y - rootNode.y;
                  const children = parentIdToChildIds[datum.id].map(childId => nodeById[childId]);
                  if (children.length > 0) {
                      children.filter(child => child.distance_to_root == datum.distance_to_root + 1).forEach(child => {
                          child.display_endabled = true;
                          if (datum !== rootNode) {
                              child.x = datum.x + xDelta;
                              child.y = datum.y + yDelta;
                          }
                      });
                      render(inputArgs);
                  }
              });

        const distanceToCenter = alpha => {
            return () => {
	        nodeData.filter(datum => datum.display_endabled).forEach(datum => {
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
	        nodeData.filter(datum => datum.display_endabled).forEach(child => {
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
	        nodeData.filter(datum => datum.display_endabled).forEach(parent => {
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

        drag.on("drag", (d,i) => {
            d.x += d3.event.dx;
            d.y += d3.event.dy;
        });

	simulation
            .force('center', d3.forceCenter(svgWidth / 2, svgHeight / 2))
            .force('links', linkForce(linkAlpha))
            .force('sibling-force', siblingForce(siblingAlpha))
            .force('distance-to-center', distanceToCenter(distanceToCenterAlpha))
            .force('collide', d3.forceCollide(paddingBetweenNodes).strength(0.5).iterations(200))
	    .nodes(nodeData.filter(datum => datum.display_endabled)).on('tick', () => {
		nodeGroup
		    .attr('cx', datum => datum.x)
		    .attr('cy', datum => datum.y)
                    .call(drag);
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
	    const nodeData = data.nodes.map(datum => Object.assign(datum, {display_endabled: datum.distance_to_root == 0}));
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
            window.removeEventListener('resize', redraw);
            redraw = () => {
                render({
                    nodeData,
                    linkData,
                    rootNode,
                    nodeById,
                    parentIdToChildIds,
                    childIdToParentids,
                    distanceToCenterFactorByDepth
                });
            };
	    redraw();
            window.addEventListener('resize', redraw);
	}).catch(err => {
	    console.error(err.message);
	    return;
	});
};

const runVisualization = () => {
    const validDatasetNames = ['ai', 'coronary_artery_disease', 'crime', 'engineer', 'financial_services', 'military_aircraft'];
    const specifiedDatasetName = window.location.hash.slice(1);
    if (validDatasetNames.includes(specifiedDatasetName)) {
        hierarchyMain(specifiedDatasetName);
    } else {
        window.location.hash='#engineer';
        location.reload();
    }
};

const runAIVisualization = () => {
    window.location.hash='#ai';
    location.reload();
};

const runCoronaryArteryDiseaseVisualization = () => {
    window.location.hash='#coronary_artery_disease';
    location.reload();
};

const runCrimeVisualization = () => {
    window.location.hash='#crime';
    location.reload();
};

const runEngineerVisualization = () => {
    window.location.hash='#engineer';
    location.reload();
};

const runFinancialServicesVisualization = () => {
    window.location.hash='#financial_services';
    location.reload();
};

const runMilitaryAircraftVisualization = () => {
    window.location.hash='#military_aircraft';
    location.reload();
};
