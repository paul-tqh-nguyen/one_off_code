
const nodes = [
    { id: "mammal", group: 0, label: "Mammals", level: 1, customY: 10 },
    { id: "dog"   , group: 0, label: "Dogs"   , level: 2, customY: 10, customY: 10 },
    { id: "cat"   , group: 0, label: "Cats"   , level: 2, customY: 10 },
    { id: "fox"   , group: 0, label: "Foxes"  , level: 2, customY: 10 },
    { id: "elk"   , group: 0, label: "Elk"    , level: 2, customY: 10 },
    { id: "insect", group: 1, label: "Insects", level: 1, customY: 10 },
    { id: "ant"   , group: 1, label: "Ants"   , level: 2, customY: 10 },
    { id: "bee"   , group: 1, label: "Bees"   , level: 2, customY: 10 },
    { id: "fish"  , group: 2, label: "Fish"   , level: 1, customY: 10 },
    { id: "carp"  , group: 2, label: "Carp"   , level: 2, customY: 10 },
    { id: "pike"  , group: 2, label: "Pikes"  , level: 2, customY: 10 }
];

const plotContainer = document.getElementById('hierarchy');
const svg = d3.select('#hierarchy-svg');
svg
    .attr('width', `${plotContainer.clientWidth}px`)
    .attr('height', `${plotContainer.clientHeight}px`);
const svg_width = parseFloat(svg.attr('width'));
const svg_height = parseFloat(svg.attr('height'));

const simulation = d3
      .forceSimulation()
      .force('center', d3.forceCenter(svg_width / 2, svg_height / 2))
      .force('charge', d3.forceManyBody().strength(-20));

const nodeGroup = svg.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", 10)
      .attr("fill", 'red');

const textElements = svg.append("g")
      .attr("class", "texts")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(node =>  node.label)
      .attr("font-size", 15)
      .attr("dx", 15)
      .attr("dy", 5);

simulation.nodes(nodes).on('tick', () => {
    nodeGroup
	.attr('cx', node => node.x)
	.attr('cy', node => node.customY);
    textElements
	.attr('x', node => node.x)
	.attr('y', node => node.y);
});
