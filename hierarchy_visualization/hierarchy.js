
const hierarchyMain = () => {
    
    const dataLocation = './hierarchy_data.json';

    const mean = inputArray => inputArray.reduce((a, b) => a + b, 0) / inputArray.length;
    
    const plotContainer = document.getElementById('hierarchy');
    const svg = d3.select('#hierarchy-svg');
    const simulation = d3.forceSimulation()
	  .alphaDecay(0.00001)
	  .velocityDecay(0.2);
    
    const chargeStrength = -160;
    const nodeRadius = 10;
    const textFontSize = 15;
    const paddingBetweenNodes = 30;
    
    const margin = {
        top: 50,
        bottom: 50,
        left: 50,
        right: 50,
    };
    
    const render = (nodeData, linkData, nodeById, parentIdToChildIds, childIdToParentids) => {
	
	svg
	    .attr('width', `${plotContainer.clientWidth}px`)
	    .attr('height', `${plotContainer.clientHeight}px`)
	    .selectAll('*')
	    .remove();
	const svgWidth = parseFloat(svg.attr('width'));
	const svgHeight = parseFloat(svg.attr('height'));

	const edgeGroup = svg.append('g')
	      .selectAll('line')
	      .data(linkData)
	      .enter().append('line')
	      .attr('stroke', 'black')
	      .attr('stroke-opacity', 1)
	      .attr('stroke-width', 3);

	const nodeGroup = svg.append('g')
	      .selectAll('circle')
	      .data(nodeData)
	      .enter().append('circle')
	      .attr('r', nodeRadius)
	      .attr('fill', 'red');
	
	const textGroup = svg.append('g')
	      .selectAll('text')
	      .data(nodeData)
	      .enter().append('text')
	      .text(datum =>  datum.distance_to_root)
	      .attr('font-size', textFontSize)
	      .attr('dx', nodeRadius + 5)
	      .attr('dy', nodeRadius / 2);

	const collide = alpha => {
	    var quadtree = d3.quadtree()
		.x(datum => datum.x)
		.y(datum => datum.y)
		.addAll(nodeData);
	    return datum => {
		const datumBoundingDistance = datum.radius + paddingBetweenNodes;
		const datumLeftX = datum.x - datumBoundingDistance;
		const datumTopY = datum.y - datumBoundingDistance;
		const datumRightX = datum.x + datumBoundingDistance;
		const datumBottomY = datum.y + datumBoundingDistance;
		quadtree.visit((quadtreeNode, quadtreeNodeLeftX, quadtreeNodeTopY, quadtreeNodeRightX, quadtreeNodeBottomY) => {
		    if (quadtreeNode.data && (quadtreeNode.data !== datum)) {
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
		    const collisionDetected = quadtreeNodeLeftX > datumRightX || quadtreeNodeRightX < datumLeftX || quadtreeNodeTopY > datumBottomY || quadtreeNodeBottomY < datumTopY;
		    return collisionDetected;
		});
	    };
	};
	
	const horizontalCenter = alpha => {
	    return () => nodeData.forEach(datum => {
		datum.x = datum.x * alpha + Math.min(svgWidth / 2) * (1-alpha);
	    });
	};
	
	const boundingBoxForce = () => {
	    nodeData.forEach(datum => {
		datum.x = Math.max(margin.left, Math.min(svgWidth - margin.right, datum.x));
		datum.y = Math.max(margin.top, Math.min(svgHeight - margin.bottom, datum.y));
	    });
	};

	// const linkPull = alpha => {
	//     nodeData.forEach(datum => {
	// 	const parentIds = childIdToParentids[datum.id];
	// 	const childIds = parentIdToChildIds[datum.id];
	// 	const parentXs = parentIds.map(parentId => {
	// 	    console.log('\n\n\n\n');
        //             console.log(`parentId ${JSON.stringify(parentId)}`);
        //             console.log(`nodeById[parentId] ${JSON.stringify(nodeById[parentId])}`);
        //             console.log(`nodeById[parentId].x ${JSON.stringify(nodeById[parentId].x)}`);
	// 	    return nodeById[parentId].x;
	// 	});
	// 	const childXs = childIds.map(childId => {
	// 	    return nodeById[childId].x;
	// 	});
	// 	const meanX = mean(parentXs.concat(childXs));
	// 	// console.log('\n\n\n\n');
        //         // console.log(`parentIds ${JSON.stringify(parentIds)}`);
        //         // console.log(`parentXs ${JSON.stringify(parentXs)}`);
        //         // console.log(`childIds ${JSON.stringify(childIds)}`);
        //         // console.log(`childXs ${JSON.stringify(childXs)}`);
        //         // console.log(`datum.x ${JSON.stringify(datum.x)}`);
	// 	// datum.x =  alpha * datum.x + (1 - alpha) * meanX;
        //         console.log(`datum.x ${JSON.stringify(datum.x)}`);
	//     });
	// };
	
	simulation
	// .force('charge', d3.forceManyBody().strength(-50))
	    .force('center', d3.forceCenter(svgWidth / 2, svgHeight / 2))
	    .force('horizontal-center', horizontalCenter(0.999))
	    .force('bounding-box', boundingBoxForce)
	// .force('link-pull', linkPull(0.01))
	    .force('y', d3.forceY().y(datum => datum.distance_to_root * nodeRadius * 20))
	    .force('x', d3.forceX().x(datum => {
	    	const parentIds = childIdToParentids[datum.id];
	    	const parentXs = parentIds ? parentIds.map(parentId => nodeById[parentId].x) : [];
	    	const childIds = parentIdToChildIds[datum.id];
	    	const childXs = childIds ? childIds.map(childId => nodeById[childId].x) : [];
	    	const meanX = mean(parentXs.concat(childXs));
	    	return meanX;
	    }).strength(0.001))
	    .nodes(nodeData).on('tick', () => {
		nodeGroup
		    .each(collide(0.5))
		    .attr('cx', datum => datum.x)
		    .attr('cy', datum => datum.y);
		edgeGroup
		    .attr('x1', datum => nodeById[datum.parent].x)
		    .attr('y1', datum => nodeById[datum.parent].y)
		    .attr('x2', datum => nodeById[datum.child].x)
		    .attr('y2', datum => nodeById[datum.child].y);
		textGroup
		    .attr('x', datum => datum.x)
		    .attr('y', datum => datum.y);
	    })
	    .restart();
    };
    
    d3.json(dataLocation)
	.then(data => {
	    const nodeData = data.nodes;
	    const linkData = data.links;
	    let nodeById = {};
	    nodeData.forEach(node => {
		nodeById[node.id] = node;
	    });
	    let parentIdToChildIds = {};
	    let childIdToParentids = {};
	    nodeData.forEach(datum => {
		parentIdToChildIds[datum.id] = [];
		childIdToParentids[datum.id] = [];
	    });
	    linkData.forEach(datum => {
		parentIdToChildIds[datum.parent].push(datum.child);
		childIdToParentids[datum.child].push(datum.parent);
	    });
            const redraw = () => render(nodeData, linkData, nodeById, parentIdToChildIds, childIdToParentids);
	    redraw();
	    window.addEventListener('resize', redraw);
	}).catch(err => {
	    console.error(err.message);
	    return;
	});
};

hierarchyMain();
