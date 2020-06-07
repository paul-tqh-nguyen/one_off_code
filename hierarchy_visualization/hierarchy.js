
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
    const paddingBetweenNodes = 10;
    
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
	    const quadtree = d3.geom.quadtree(nodes);
	    return datum => {
		const r = datum.radius + paddingBetweenNodes;
		const nx0 = datum.x - r;
		const ny0 = datum.y - r;
		const nx1 = datum.x + r;
		const ny1 = datum.y + r;
		quadtree.visit((quadtreeNode, quadtreeNodeLeftX, quadtreeNodeTopY, quadtreeNodeRightX, quadtreeNodeBottomY) => {
		    if (quadtreeNode.point && (quadtreeNode.point !== datum)) {
			let xDelta = datum.x - quadtreeNode.point.x;
			let yDelta = datum.y - quadtreeNode.point.y;
			let distance = Math.sqrt(xDelta * xDelta + yDelta * yDelta);
			let minimumDistance = datum.radius + quadtreeNode.point.radius + paddingBetweenNodes;
			if (distance < minimumDistance) {
			    distance = (distance - minimumDistance) / distance * alpha;
			    xDelta *= distance;
			    datum.x -= xDelta;
			    yDelta *= distance;
			    datum.y -= yDelta;
			    quadtreeNode.point.x += xDelta;
			    quadtreeNode.point.y += yDelta;
			}
		    }
		    const collisionDetected = quadtreeNodeLeftX > nx1 || quadtreeNodeRightX < nx0 || quadtreeNodeTopY > ny1 || quadtreeNodeBottomY < ny0;
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
