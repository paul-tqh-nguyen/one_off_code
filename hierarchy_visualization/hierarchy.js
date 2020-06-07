
const nodes = [
    { id: "mammal", group: 0, label: "Mammals", level: 1 },
    { id: "dog"   , group: 0, label: "Dogs"   , level: 2 },
    { id: "cat"   , group: 0, label: "Cats"   , level: 2 },
    { id: "fox"   , group: 0, label: "Foxes"  , level: 2 },
    { id: "elk"   , group: 0, label: "Elk"    , level: 2 },
    { id: "insect", group: 1, label: "Insects", level: 1 },
    { id: "ant"   , group: 1, label: "Ants"   , level: 2 },
    { id: "bee"   , group: 1, label: "Bees"   , level: 2 },
    { id: "fish"  , group: 2, label: "Fish"   , level: 1 },
    { id: "carp"  , group: 2, label: "Carp"   , level: 2 },
    { id: "pike"  , group: 2, label: "Pikes"  , level: 2 }
];

const hierarchyMain = () => {
    
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
    
    const redraw = () => {
	
	svg
	    .attr('width', `${plotContainer.clientWidth}px`)
	    .attr('height', `${plotContainer.clientHeight}px`)
	    .selectAll("*")
	    .remove();;
	const svg_width = parseFloat(svg.attr('width'));
	const svg_height = parseFloat(svg.attr('height'));
	
 	const collide = alpha => {
	    var quadtree = d3.quadtree()
		.x(node => node.x)
		.y(node => node.y)
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
	    nodes.forEach(node => {
		node.x = Math.max(margin.left, Math.min(svg_width - margin.right, node.x));
		node.y = Math.max(margin.top, Math.min(svg_height - margin.bottom, node.y));
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
	      .text(node =>  node.label)
	      .attr("font-size", textFontSize)
	      .attr("dx", nodeRadius + 5)
	      .attr("dy", nodeRadius / 2);

	simulation
	// .force('y', d3.forceY().y(datum => 10))
	    .force('center', d3.forceCenter(svg_width / 2, svg_height / 2))
	    .force('charge', d3.forceManyBody().strength(chargeStrength))
	    .force('bounding-box', boundingBoxForce)
	    .nodes(nodes).on('tick', () => {
		nodeGroup
		    .each(collide(0.5))
		    .attr('cx', node => node.x)
		    .attr('cy', node => node.y);
		textGroup
		    .attr('x', node => node.x)
		    .attr('y', node => node.y);
	    })
	    .restart();
    };
    
    redraw();
    window.addEventListener('resize', redraw);
};

hierarchyMain();
