
const hierarchyMain = () => {
    
    const dataLocation = './hierarchy_data.json';
    
    const plotContainer = document.getElementById('hierarchy');
    const svg = d3.select('#hierarchy-svg');
    const simulation = d3.forceSimulation();
    
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
    
    const render = (nodes) => {
	
	svg
	    .attr('width', `${plotContainer.clientWidth}px`)
	    .attr('height', `${plotContainer.clientHeight}px`)
	    .selectAll("*")
	    .remove();
	const svg_width = parseFloat(svg.attr('width'));
	const svg_height = parseFloat(svg.attr('height'));

	const collide = alpha => {
	    var quadtree = d3.quadtree()
		.x(datum => datum.x)
		.y(datum => datum.y)
		.addAll(nodes);
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
	
	const boundingBoxForce = () => {
	    nodes.forEach(datum => {
		datum.x = Math.max(margin.left, Math.min(svg_width - margin.right, datum.x));
		datum.y = Math.max(margin.top, Math.min(svg_height - margin.bottom, datum.y));
	    });
	};
	
	const nodeGroup = svg.append("g")
	      .selectAll("circle")
	      .data(nodes)
	      .enter().append("circle")
	      .attr("r", nodeRadius)
	      .attr("fill", 'red');
	
	const textGroup = svg.append("g")
	      .selectAll("text")
	      .data(nodes)
	      .enter().append("text")
	      .text(datum =>  `${datum.label} (${datum.distance_to_root})`)
	      .attr("font-size", textFontSize)
	      .attr("dx", nodeRadius + 5)
	      .attr("dy", nodeRadius / 2);
	   
	simulation
	    .force('y', d3.forceY().y(datum => datum.distance_to_root*100))
	    .force('center', d3.forceCenter(svg_width / 2, svg_height / 2))
	    .force('charge', d3.forceManyBody().strength(chargeStrength))
	    .force('bounding-box', boundingBoxForce)
	    .nodes(nodes).on('tick', () => {
		nodeGroup
		    .each(collide(0.5))
		    .attr('cx', datum => datum.x)
		    .attr('cy', datum => datum.y);
		textGroup
		    .attr('x', datum => datum.x)
		    .attr('y', datum => datum.y);
	    })
	    .restart();
    };

    d3.json(dataLocation)
	.then(data => {
	    const nodes = data.nodes.slice(0,50);
            const redraw = () => render(nodes);
	    redraw();
	    window.addEventListener('resize', redraw);
	}).catch(err => {
	    console.error(err.message);
	    return;
	});
};

hierarchyMain();
