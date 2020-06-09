
const hierarchyMain = () => {
    
    const dataLocation = './hierarchy_data.json';

    const mean = inputArray => inputArray.reduce((a, b) => a + b, 0) / inputArray.length;
    const shuffle = (inputArray) => inputArray.sort(() => Math.random() - 0.5);

    const plotContainer = document.getElementById('hierarchy');
    const svg = d3.select('#hierarchy-svg');
    
    const alphaDecay = 0.01;
    const velocityDecay = 0.00001;
    const collisionAlpha = 0.5;
    const distanceToCenterAlpha = 1.0;
    const linkAlpha = 0.25;
    const siblingAlpha = 0.5;

    const nodeRadius = 10;
    const edgeWidth = 1;
    const hiddenEdgeWidth = 8;
    const paddingBetweenNodes = 20;
    const approximateCircumferenceDistancePerNode = 15;
    const margin = {
        top: 100,
        bottom: 100,
        left: 100,
        right: 100,
    };

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

        const svgContent = svg.append('g');
        svg.call(d3.zoom().on('zoom', () => {
            svgContent.attr('transform', d3.event.transform);
        }));
	const edgeGroup = svgContent.append('g')
	      .selectAll('line')
	      .data(linkData)
	      .enter().append('line')
	      .attr('stroke', 'black')
	      .attr('stroke-opacity', 1)
	      .attr('stroke-width', 3);

	const nodeGroup = svgContent.append('g')
	      .selectAll('circle')
	      .data(nodeData)
	      .enter().append('circle')
	      .attr('r', nodeRadius)
	      .attr('fill', 'red');

	const collide = alpha => {
	    var quadtree = d3.quadtree()
		.x(datum => datum.x)
		.y(datum => datum.y)
		.addAll(nodeData);
	    return datum => {
		quadtree.visit((quadtreeNode, quadtreeNodeLeftX, quadtreeNodeTopY, quadtreeNodeRightX, quadtreeNodeBottomY) => {
                    let goalBetweenNodes = paddingBetweenNodes;
		    if (quadtreeNode.data && (quadtreeNode.data !== datum)) {
                        if (datum.distance_to_root == quadtreeNode.data.distance_to_root) {
                            goalBetweenNodes *= 2;
                        }
			let xDelta = datum.x - quadtreeNode.data.x;
			let yDelta = datum.y - quadtreeNode.data.y;
			let distance = Math.sqrt(xDelta * xDelta + yDelta * yDelta);
			let minimumDistance = nodeRadius + nodeRadius + paddingBetweenNodes;
			if (distance < minimumDistance) {
			    distance = (distance - minimumDistance) / distance * alpha;
			    xDelta *= distance;
			    datum.x -= xDelta;
			    yDelta *= distance;
			    datum.y -= yDelta;
			    quadtreeNode.data.x += xDelta;
			    quadtreeNode.data.y += yDelta;
			}
		    }
		    const datumBoundingDistance = datum.radius + goalBetweenNodes;
		    const datumLeftX = datum.x - datumBoundingDistance;
		    const datumTopY = datum.y - datumBoundingDistance;
		    const datumRightX = datum.x + datumBoundingDistance;
		    const datumBottomY = datum.y + datumBoundingDistance;
		    const collisionDetected = quadtreeNodeLeftX > datumRightX || quadtreeNodeRightX < datumLeftX || quadtreeNodeTopY > datumBottomY || quadtreeNodeBottomY < datumTopY;
		    return collisionDetected;
		});
	    };
	};

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
	    .nodes(nodeData).on('tick', () => {
		nodeGroup
		    .each(collide(collisionAlpha))
		    .attr('cx', datum => datum.x)
		    .attr('cy', datum => datum.y);
		edgeGroup
		    .attr('stroke-opacity', datum => {
                        return nodeById[datum.child].distance_to_root - nodeById[datum.parent].distance_to_root > 1 ? 0.05 : 1;
                    })
		    .attr('stroke-width', datum => {
                        return nodeById[datum.child].distance_to_root - nodeById[datum.parent].distance_to_root > 1 ? hiddenEdgeWidth : edgeWidth;
                    })
		    .attr('x1', datum => nodeById[datum.parent].x)
		    .attr('y1', datum => nodeById[datum.parent].y)
		    .attr('x2', datum => nodeById[datum.child].x)
		    .attr('y2', datum => nodeById[datum.child].y);
	    })
	    .restart();
    };

    const generateDistanceToCenterFactorByDepth = nodeData => {
        let nodesPerDepth = {};
	nodeData.forEach(node => {
            if (node.distance_to_root in nodesPerDepth) {
                nodesPerDepth[node.distance_to_root] += 1;
            } else {
                nodesPerDepth[node.distance_to_root] = 1;
            }
        });
        const distanceToCenterFactorByDepth = {};
        Object.keys(nodesPerDepth).forEach(depth => {
            const nodeCount = nodesPerDepth[depth];
            const approximateCircumference = nodeCount * approximateCircumferenceDistancePerNode;
            const expectedRadius = approximateCircumference / (2 * Math.PI);
            distanceToCenterFactorByDepth[depth] = Math.max(nodeRadius * 20, expectedRadius);
        });
        let currentDistanceFromRoot = 0;
        Object.keys(distanceToCenterFactorByDepth).sort().map(depth => {
            distanceToCenterFactorByDepth[depth] += currentDistanceFromRoot;
            currentDistanceFromRoot = distanceToCenterFactorByDepth[depth];
        });
        return distanceToCenterFactorByDepth;
    };
    
    d3.json(dataLocation)
	.then(data => {
	    const nodeData = data.nodes;
	    const linkData = data.links;
            const rootNode = nodeData.filter(datum => datum.distance_to_root == 0)[0];
	    let nodeById = {};
	    let parentIdToChildIds = {};
	    let childIdToParentids = {};
	    nodeData.forEach(node => {
		nodeById[node.id] = node;
		parentIdToChildIds[node.id] = [];
		childIdToParentids[node.id] = [];
	    });
	    linkData.forEach(datum => {
		parentIdToChildIds[datum.parent].push(datum.child);
		childIdToParentids[datum.child].push(datum.parent);
	    });
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
